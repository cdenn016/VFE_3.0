r"""Conventional pre-LN transformer baseline for the conventional rows of Table M2.

``GL(K)_attention_v2`` Table M2 (``tab:placeholder_main_results``) compares the evaluated realization
with conventional transformers trained under the same tokenizer, context, stride and budget. This
module holds the whole baseline: :class:`BaselineConfig` (with :meth:`BaselineConfig.matched_to`,
which derives the matched arm from any VFE base config), :class:`BaselineTransformer`, its optimizer
(:func:`build_optimizer`), its parameter-role map (:func:`baseline_parameter_role`), and the
compute-matching helpers (:func:`steps_for_device_hours`, :func:`width_ladder_rule`,
:func:`calibrated_tokens_per_s`, :func:`measured_train_device_hours`).

Architecture. With token ids :math:`x_i`, width :math:`K`, :math:`H` heads of width
:math:`d = K/H`, :math:`L` layers and MLP width :math:`eK`, every layer applies

.. math::
    u_i = \mathrm{LN}_1(h_i),\qquad
    \tilde u_i = A_{x_i} u_i,\quad A_v = I + X_v = \bigoplus_{h=1}^{H} (I_d + X_{v,h}),

    \beta^h_{ij} = \operatorname{softmax}_j\Big(
        \frac{(R_i W_Q \tilde u_i)_h^\top (R_j W_K \tilde u_j)_h}{\sqrt d} + B_{h,ij}\Big),\qquad
    h_i \leftarrow h_i + W_O \bigoplus_h \sum_j \beta^h_{ij} (W_V \tilde u_j)_h,

    h_i \leftarrow h_i + W_2\, \mathrm{GELU}(W_1 \mathrm{LN}_2(h_i) + b_1) + b_2,

with :math:`h^{(0)}_i = E_{x_i}` (untied token table), :math:`B` the VFE's own attention log-prior
(:func:`vfe3.attention_prior.attention_log_prior`, causal ALiBi by default), :math:`R_i` the identity, or
under ``pos_rotation='rope'`` the per-head rotary map :math:`R_i = \bigoplus_h R(i\theta)` of
:func:`vfe3.geometry.rope.build_rope_rotation` (pairs :math:`(2k, 2k+1)` of each head rotate by
:math:`i\,b^{-2k/d}`, base :math:`b` = ``rope_base``), so that
:math:`(R_i q)^\top R_j k = q^\top R_{j-i} k` depends on position only through :math:`j - i`. These are
the VFE's comparison-space RoPE frequencies (a comparison ``rope_mode``), which rotate the projected
query and key moments of each head by the same :math:`R_i`. Exact (erf) GELU, no
dropout and bias-free attention maps. The per-type read map :math:`A_v` exists only with
``per_type_table='block_read'``; it is one ``(V, H, d, d)`` table shared by every layer, applied to the
attention input, block-diagonal on the heads, so it has exactly the size of the VFE token-frame table
:math:`V n_{\mathrm{gen}} = V K d` and acts on context-dependent state (it cannot be absorbed into
:math:`E`). ``per_type_table='block_rw'`` uses the same table the way the VFE uses its token frames, on
both sides of attention: with :math:`A_v = \exp(X_v)` blockwise, the attention input is read through
:math:`A_{x_i}^{-1}` and the attention output is written through :math:`A_{x_i}` before the residual
add, the analogue of :math:`U_i^{-1}` before the comparison and content maps and :math:`U_i` after the
receiving map; zero initialization gives the plain baseline at step 0.

``norm_placement='post_shared'`` replaces the pre-LN layout by the realization's: one LayerNorm shared
by every layer, applied after the attention residual and before the MLP residual,

.. math::
    h_i \leftarrow \mathrm{LN}\big(h_i + \mathrm{Attn}(h)_i\big),\qquad
    h_i \leftarrow h_i + W_2\, \mathrm{GELU}(W_1 h_i + b_1) + b_2,

with no final normalization (the VFE's ``norm_type_final='none'``); the context-processing count is then
exactly the VFE's. With ``attention_residual=False`` (copied by :meth:`BaselineConfig.matched_to` from a
VFE base without an attention residual) the first update is
:math:`h_i \leftarrow \mathrm{LN}(\mathrm{Attn}(h)_i)`, as that base computes. The decoder is the VFE linear decoder,

.. math::
    \mathrm{logits}_i = W_{\mathrm{out}} \mathrm{LN}_f(h^{(L)}_i) + b_{\mathrm{out}} + \kappa \log\pi,

with :math:`\log\pi` the add-one-smoothed training unigram log-frequencies, evaluated by the same fused
chunked fp32 cross-entropy (:meth:`vfe3.model.prior_bank.PriorBank.decode_ce_linear_chunked`). Blocks run
under the configured autocast (``amp_dtype``); normalizations, the softmax, the decode and the CE run
in fp32. The attention softmax is explicit (no SDPA), so ``torch.use_deterministic_algorithms`` holds.

Parameter counts. Context processing (every attention, MLP and normalization parameter),

.. math::
    P_{\mathrm{ctx}} = L\,(4K^2 + 2eK^2 + eK + K + 4K) + 2K,

which is the VFE count :math:`L(4K^2 + 2eK^2 + eK + K) + 2K` plus :math:`4KL` (per-sublayer
normalizations; the VFE shares one), and exactly the VFE count under ``norm_placement='post_shared'``.
Per-type tables: :math:`2VK + V` (:math:`E`, :math:`W_{\rm out}`,
:math:`b_{\rm out}`), plus :math:`VKd` with the read map. No positional parameters (ALiBi and RoPE
are both parameter-free).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from vfe3.contracts import VFE_MODEL_FAMILY, model_family_of
from vfe3.model.prior_bank import DecodeCEResult, PriorBank

MODEL_FAMILY: str = "conventional_transformer"
PER_TYPE_TABLES: Tuple[str, ...] = ("none", "block_read", "block_rw")
NORM_PLACEMENTS: Tuple[str, ...] = ("pre", "post_shared")
PER_TYPE_TABLE_INITS: Tuple[str, ...] = ("zero", "phi_like")
ATTENTION_PRIORS: Tuple[str, ...] = ("causal_alibi", "causal")
POS_ROTATIONS: Tuple[str, ...] = ("none", "rope")
AMP_DTYPES: Tuple[Optional[str], ...] = (None, "bf16", "fp16")
GRAD_CLIP_SCOPES: Tuple[str, ...] = ("role", "group")
FP32_MATMUL_PRECISIONS: Tuple[Optional[str], ...] = (None, "highest", "high", "medium")
DECODE_CE_CHECKPOINTS: Tuple[str, ...] = ("auto", "auto_aggregate", "always", "never")

# Pre-declared compute-matching rule (Table M2, "matched measured compute"). Hoffmann et al. (2022,
# arXiv:2203.15556) put the compute-optimal token count near 20 tokens per parameter; Muennighoff et
# al. (2023, arXiv:2305.16264) find repeated data worth close to fresh data up to about 4 epochs and
# rapidly diminishing past roughly 16, so the token requirement is capped at MAX_PASSES passes over
# the unique training tokens.
TOKENS_PER_CONTEXT_PARAMETER: float = 20.0
MAX_PASSES: float = 16.0

# Fields that the matched arm copies from the VFE base config (identical names on both sides).
MATCHED_FIELDS: Tuple[str, ...] = (
    "vocab_size", "embed_dim", "n_heads", "n_layers", "max_seq_len", "batch_size", "eval_stride",
    "max_steps", "warmup_steps", "min_lr_frac", "grad_clip", "grad_clip_scope", "grad_accum_steps",
    "train_window_offset", "amp_dtype", "fp32_matmul_precision", "seed", "deterministic", "fill_uninitialized_memory",
    "log_interval", "eval_interval",
    "checkpoint_interval", "val_diagnostic_batches", "beta_attention_prior", "alibi_slope",
    "decode_unigram_prior", "unigram_kappa", "decode_chunk_size", "decode_ce_checkpoint",
)

_INT_FIELDS: Tuple[str, ...] = (
    "vocab_size", "embed_dim", "n_heads", "n_layers", "mlp_expansion", "max_seq_len", "batch_size",
    "max_steps", "warmup_steps", "grad_accum_steps", "seed", "log_interval", "eval_interval",
    "checkpoint_interval", "val_diagnostic_batches", "decode_chunk_size",
)
_FLOAT_FIELDS: Tuple[str, ...] = (
    "lr", "weight_decay", "adam_beta1", "adam_beta2", "adam_eps", "min_lr", "min_lr_frac",
    "init_std", "layernorm_eps", "alibi_slope", "rope_base", "rope_fraction", "unigram_kappa",
    "per_type_table_init_scale", "hidden_lr_scale",
)
_OPTIONAL_FLOAT_FIELDS: Tuple[str, ...] = ("grad_clip", "per_type_table_lr", "per_type_table_weight_decay")
_BOOL_FIELDS: Tuple[str, ...] = (
    "deterministic", "fill_uninitialized_memory", "trust_resume_checkpoint", "generate_embedding_figures",
    "decode_bias",
    "decode_unigram_prior", "attention_residual",
)
_WRAPPER_PREFIXES: Tuple[str, ...] = ("_orig_mod.", "module.")


def _derived_seed(seed: int, *stream: object) -> int:
    r"""Deterministic 63-bit generator seed for one named init stream of ``seed`` (BLAKE2b, unsalted)."""
    key = repr((int(seed), *stream)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") & (2 ** 63 - 1)


# ----------------------------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------------------------

@dataclass
class BaselineConfig:
    r"""Configuration of :class:`BaselineTransformer` and its training protocol.

    Every field the shared loop (:func:`vfe3.train.train`), :func:`vfe3.run_artifacts.finalize_run` or
    the paper assemblers read keeps its :class:`vfe3.config.VFE3Config` name. ``model_family`` is the
    serialized dispatch tag: every loader builds this config and :class:`BaselineTransformer` when a
    saved config carries ``model_family='conventional_transformer'``. Derive the matched arm from a VFE
    base with :meth:`matched_to`, never by hand.
    """

    model_family:          str           = MODEL_FAMILY
    # Shape.
    vocab_size:            int           = 50257
    embed_dim:             int           = 64          # K (residual width)
    n_heads:               int           = 2           # H; head width d = K / H
    n_layers:              int           = 2           # L
    mlp_expansion:         int           = 8           # e; MLP hidden width e K
    max_seq_len:           int           = 128         # N (context of training windows)
    # Data protocol.
    batch_size:            int           = 64
    eval_stride:           Optional[int] = None        # validation/test sliding-window stride
    train_window_offset:   str           = "fixed"     # "fixed" | "per_epoch_random" (VFE3Config's)
    # Optimization (AdamW; warmup then half-cosine, floored as in vfe3.train._floor_lr_lambdas).
    max_steps:             int           = 15000
    warmup_steps:          int           = 100
    lr:                    float         = 6e-4
    weight_decay:          float         = 0.1         # decoupled; matrices and tables only
    adam_beta1:            float         = 0.9
    adam_beta2:            float         = 0.95
    adam_eps:              float         = 1e-8
    # Multiplier of lr on the hidden matrices W_{Q,K,V,O}, W_{1,2} only (muP-style width transfer, Yang et al.
    # 2022, arXiv:2203.03466; total_parameter_width sets K_base / K). 1.0 keeps one decay group.
    hidden_lr_scale:       float         = 1.0
    min_lr:                float         = 0.0         # retired 2026-10-02: only 0 (kept for saved configs)
    min_lr_frac:           float         = 0.01
    grad_clip:             Optional[float] = 1.0
    # "role" | "group": one L2 clip per optimizer role (mu, and phi with a per-type table) or per optimizer
    # group, read by the shared loop as VFE3Config.grad_clip_scope is.
    grad_clip_scope:       str           = "role"
    grad_accum_steps:      int           = 1
    init_std:              float         = 0.02        # N(0, init_std^2) matrices; W_O and W_2 / sqrt(2L)
    layernorm_eps:         float         = 1e-5
    # Precision and runtime.
    amp_dtype:             Optional[str] = None        # None | "bf16" | "fp16" (blocks only)
    fp32_matmul_precision: Optional[str] = "highest"
    seed:                  int           = 6
    deterministic:         bool          = True
    fill_uninitialized_memory: bool      = True        # read only under deterministic algorithms; matched
    # Cadence and resume.
    log_interval:          int           = 100
    eval_interval:         int           = 1500
    checkpoint_interval:   int           = 0
    val_diagnostic_batches: int          = 1
    resume_from:           Optional[str] = None
    trust_resume_checkpoint: bool        = False
    generate_embedding_figures: bool     = False       # the VFE embedding figures do not apply
    # Normalization layout: "pre" (per-sublayer pre-LN plus a final LN) | "post_shared" (the VFE's one
    # shared LayerNorm after the attention residual, no final LN).
    norm_placement:        str           = "pre"
    # post_shared only: False drops the attention residual, LN(Attn(h)) instead of LN(h + Attn(h)), as a
    # VFE base with attention_residual=False computes.
    attention_residual:    bool          = True
    # Attention prior (the VFE's registry).
    beta_attention_prior:  str           = "causal_alibi"
    alibi_slope:           float         = 1.0
    # "none" | "rope": rotary query/key positions per head (the VFE's comparison-space RoPE).
    pos_rotation:          str           = "none"
    rope_base:             float         = 100.0       # rope only: frequency base
    rope_fraction:         float         = 1.0         # rope only: p-RoPE fraction of rotated pairs per head
    # Decode (the VFE linear decoder).
    decode_mode:           str           = "linear"
    decode_bias:           bool          = True
    decode_unigram_prior:  bool          = True
    unigram_kappa:         float         = 1.0
    decode_chunk_size:     int           = 8192
    decode_ce_checkpoint:  str           = "auto"
    # Per-type table of the frame table's size (Table M2): "block_read" reads through A_v = I + X_v;
    # "block_rw" reads through exp(-X_v) and writes the attention output through exp(X_v).
    per_type_table:        str           = "none"      # "none" | "block_read" | "block_rw"
    per_type_table_init:   str           = "zero"      # "zero" (step 0 = the plain baseline) | "phi_like"
    per_type_table_init_scale: float     = 0.01        # std of X under "phi_like" (the VFE phi_scale)
    per_type_table_lr:     Optional[float] = None      # None -> lr
    per_type_table_weight_decay: Optional[float] = None   # None -> weight_decay (decays A toward I)

    def __post_init__(self) -> None:
        if self.model_family != MODEL_FAMILY:
            raise ValueError(f"model_family must be {MODEL_FAMILY!r}, got {self.model_family!r}")
        for name in _INT_FIELDS:
            value = getattr(self, name)
            if type(value) is not int:
                raise ValueError(f"{name} must be an int, got {value!r}")
        for name in _BOOL_FIELDS:
            value = getattr(self, name)
            if type(value) is not bool:
                raise ValueError(f"{name} must be a bool, got {value!r}")
        for name in _FLOAT_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number, got {value!r}")
            setattr(self, name, float(value))
        for name in _OPTIONAL_FLOAT_FIELDS:
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be None or a finite number, got {value!r}")
            setattr(self, name, float(value))
        for name in ("vocab_size", "embed_dim", "n_heads", "n_layers", "mlp_expansion", "max_seq_len",
                     "batch_size", "max_steps", "grad_accum_steps", "val_diagnostic_batches",
                     "decode_chunk_size"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        for name in ("warmup_steps", "log_interval", "eval_interval", "checkpoint_interval", "seed"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        if self.embed_dim % self.n_heads:
            raise ValueError(f"embed_dim={self.embed_dim} must be divisible by n_heads={self.n_heads}")
        if self.eval_stride is not None and (type(self.eval_stride) is not int
                                             or not 1 <= self.eval_stride <= self.max_seq_len):
            raise ValueError(f"eval_stride must be None or an int in [1, max_seq_len], got {self.eval_stride!r}")
        from vfe3.data.datasets import WINDOW_OFFSET_MODES
        if self.train_window_offset not in WINDOW_OFFSET_MODES:
            raise ValueError(f"train_window_offset must be one of {WINDOW_OFFSET_MODES}, "
                             f"got {self.train_window_offset!r}")
        if self.batch_size % self.grad_accum_steps:
            raise ValueError(f"grad_accum_steps={self.grad_accum_steps} must divide batch_size={self.batch_size}")
        if self.lr <= 0.0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.hidden_lr_scale <= 0.0:
            raise ValueError(f"hidden_lr_scale must be positive, got {self.hidden_lr_scale}")
        if self.weight_decay < 0.0 or not 0.0 <= self.min_lr_frac <= 1.0:
            raise ValueError("weight_decay must be >= 0 and min_lr_frac in [0, 1]")
        if self.min_lr != 0.0:
            # The shared scheduler floor is min_lr_frac * base since 2026-10-02; the absolute
            # floor is no longer read, so a nonzero value would be silently ignored.
            raise ValueError(f"min_lr is retired (only 0 is accepted), got {self.min_lr!r}")
        if not (0.0 <= self.adam_beta1 < 1.0 and 0.0 <= self.adam_beta2 < 1.0 and self.adam_eps > 0.0):
            raise ValueError("adam_beta1, adam_beta2 must lie in [0, 1) and adam_eps must be positive")
        if self.grad_clip is not None and self.grad_clip < 0.0:
            raise ValueError(f"grad_clip must be None or >= 0, got {self.grad_clip}")
        if self.grad_clip_scope not in GRAD_CLIP_SCOPES:
            raise ValueError(f"grad_clip_scope must be one of {GRAD_CLIP_SCOPES}, got {self.grad_clip_scope!r}")
        if self.init_std <= 0.0 or self.layernorm_eps <= 0.0:
            raise ValueError("init_std and layernorm_eps must be positive")
        if self.amp_dtype not in AMP_DTYPES:
            raise ValueError(f"amp_dtype must be one of {AMP_DTYPES}, got {self.amp_dtype!r}")
        if self.fp32_matmul_precision not in FP32_MATMUL_PRECISIONS:
            raise ValueError(f"fp32_matmul_precision must be one of {FP32_MATMUL_PRECISIONS}, "
                             f"got {self.fp32_matmul_precision!r}")
        if self.beta_attention_prior not in ATTENTION_PRIORS:
            raise ValueError(f"beta_attention_prior must be one of {ATTENTION_PRIORS} (a causal prior "
                             f"with a self key), got {self.beta_attention_prior!r}")
        if self.pos_rotation not in POS_ROTATIONS:
            raise ValueError(f"pos_rotation must be one of {POS_ROTATIONS}, got {self.pos_rotation!r}")
        if not (math.isfinite(self.rope_base) and self.rope_base > 1.0):
            raise ValueError(f"rope_base must be a finite value > 1, got {self.rope_base!r}")
        if not 0.0 < self.rope_fraction <= 1.0:
            raise ValueError(f"rope_fraction must be in (0, 1], got {self.rope_fraction!r}")
        if self.decode_mode != "linear":
            raise ValueError(f"the baseline decodes linearly; decode_mode must be 'linear', got {self.decode_mode!r}")
        if self.decode_ce_checkpoint not in DECODE_CE_CHECKPOINTS:
            raise ValueError(f"decode_ce_checkpoint must be one of {DECODE_CE_CHECKPOINTS}, "
                             f"got {self.decode_ce_checkpoint!r}")
        if self.norm_placement not in NORM_PLACEMENTS:
            raise ValueError(f"norm_placement must be one of {NORM_PLACEMENTS}, got {self.norm_placement!r}")
        if not self.attention_residual and self.norm_placement != "post_shared":
            raise ValueError("attention_residual=False is defined only for norm_placement='post_shared'")
        if self.per_type_table not in PER_TYPE_TABLES:
            raise ValueError(f"per_type_table must be one of {PER_TYPE_TABLES}, got {self.per_type_table!r}")
        if self.per_type_table_init not in PER_TYPE_TABLE_INITS:
            raise ValueError(f"per_type_table_init must be one of {PER_TYPE_TABLE_INITS}, "
                             f"got {self.per_type_table_init!r}")
        if self.per_type_table_init_scale < 0.0:
            raise ValueError("per_type_table_init_scale must be >= 0")
        if self.per_type_table_lr is not None and self.per_type_table_lr < 0.0:
            raise ValueError("per_type_table_lr must be None or >= 0")
        if self.per_type_table_weight_decay is not None and self.per_type_table_weight_decay < 0.0:
            raise ValueError("per_type_table_weight_decay must be None or >= 0")
        if self.resume_from is not None and not isinstance(self.resume_from, str):
            raise ValueError("resume_from must be None or a path string")

    # -- derived sizes -------------------------------------------------------------------------

    @property
    def d_head(self) -> int:
        r"""Head width :math:`d = K / H`."""
        return self.embed_dim // self.n_heads

    def context_processing_parameters(self) -> int:
        r"""Analytic :math:`P_{\mathrm{ctx}} = L(4K^2 + 2eK^2 + eK + K + 4K) + 2K` (pre-LN), or the VFE
        count :math:`L(4K^2 + 2eK^2 + eK + K) + 2K` under ``norm_placement='post_shared'``."""
        K, e, L = self.embed_dim, self.mlp_expansion, self.n_layers
        norms = 4 * K if self.norm_placement == "pre" else 0
        return L * (4 * K * K + 2 * e * K * K + e * K + K + norms) + 2 * K

    def per_type_parameters(self) -> Dict[str, int]:
        r"""Per-type table sizes: token table :math:`VK`, decoder :math:`VK (+V)`, read map :math:`VKd`."""
        V, K = self.vocab_size, self.embed_dim
        return {
            "token_embed": V * K,
            "decode": V * K + (V if self.decode_bias else 0),
            "type_table": (V * K * self.d_head) if self.per_type_table != "none" else 0,
        }

    def total_parameters(self) -> int:
        r"""Every trainable parameter (context processing plus per-type tables)."""
        return self.context_processing_parameters() + sum(self.per_type_parameters().values())

    # -- serialization -------------------------------------------------------------------------

    def serialize(self) -> Dict[str, Any]:
        r"""The complete config as plain JSON/``weights_only``-safe values (carries ``model_family``)."""
        return asdict(self)

    @classmethod
    def from_serialized(
        cls,
        payload: Mapping[str, Any],

        *,
        source: str = "serialized baseline config",
    ) -> "BaselineConfig":
        r"""Strict inverse of :meth:`serialize`: unknown fields and a foreign family raise ``ValueError``."""
        if not isinstance(payload, Mapping):
            raise ValueError(f"{source}: a serialized baseline config must be a mapping")
        family = payload.get("model_family")
        if family != MODEL_FAMILY:
            raise ValueError(f"{source}: model_family must be {MODEL_FAMILY!r}, got {family!r}")
        known = {f.name for f in fields(cls)}
        unknown = sorted(str(key) for key in payload if key not in known)
        if unknown:
            raise ValueError(f"{source}: undeclared baseline config field(s) {unknown}")
        try:
            return cls(**dict(payload))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{source}: {exc}") from exc

    def inactive_fields(self) -> Dict[str, str]:
        r"""Fields no active path reads under this config, with the reason (the VFE3Config analog)."""
        out: Dict[str, str] = {}
        if self.per_type_table == "none":
            for name in ("per_type_table_init", "per_type_table_init_scale", "per_type_table_lr",
                         "per_type_table_weight_decay"):
                out[name] = "per_type_table='none' builds no per-type read map"
        elif self.per_type_table_init == "zero":
            out["per_type_table_init_scale"] = "per_type_table_init='zero' draws no random table"
        if self.amp_dtype is None:
            out["amp_dtype"] = "autocast is off (fp32 blocks)"
        if not self.grad_clip:
            out["grad_clip_scope"] = "only read when grad_clip > 0"
        if self.eval_stride is None:
            out["eval_stride"] = "validation and test use disjoint max_seq_len windows"
        if not self.decode_unigram_prior:
            out["unigram_kappa"] = "decode_unigram_prior=False adds no unigram prior"
        if self.pos_rotation == "none":
            out["rope_base"] = "only read by pos_rotation='rope'"
            out["rope_fraction"] = "only read by pos_rotation='rope'"
        if self.norm_placement == "pre":
            out["attention_residual"] = "only read by norm_placement='post_shared'"
        return out

    # -- matched arm ---------------------------------------------------------------------------

    @classmethod
    def matched_to(
        cls,
        base: Any,

        *,
        per_type_table: str = "none",
        **overrides: Any,
    ) -> "BaselineConfig":
        r"""The conventional arm matched to a VFE base config (a ``VFE3Config`` or a serialized mapping).

        Copied from the base: tokenizer size, width :math:`K`, heads :math:`H` (head width
        :math:`d = K/H`, the VFE block size), depth :math:`L`, context, batch, evaluation stride, step
        budget, warmup and learning-rate floors, gradient clip, its scope and accumulation, the training-window
        origin (``train_window_offset``), precision, seed,
        cadence, the attention log-prior and its slope, and the unigram decode prior
        (:data:`MATCHED_FIELDS`). A base with a ``rope_mode`` other than ``'none'`` gives the arm rotary query/key
        positions at the base's ``rope_base``; for a comparison ``rope_mode`` this is the rotation the base
        applies to its projected routing moments, while frame-space RoPE (absolute, folded into the frame)
        has no exact dense counterpart and is matched by the same relative rotation. The MLP width is the base's ``block_mlp_expansion`` times :math:`K`
        (the template router's template count :math:`eK` when the base uses one). The decode is the
        linear decoder with bias; a base with a Gaussian decode keeps its unigram prior. The per-type
        read map, when requested, has the base's token-frame size :math:`V K d`, and its ``phi_like``
        initialization scale is the base's ``phi_scale``. The peak learning rate, weight decay and
        Adam moments are baseline fields (tune them on validation); ``overrides`` sets any field
        last. Under ``norm_placement='post_shared'`` the arm copies the base's ``attention_residual``
        unless ``overrides`` sets it. A mapping may be the whole ``config.json`` record (its ``"config"`` entry is used);
        fields absent from an older mapping take the current ``VFE3Config`` defaults. The base must
        use the current full-width projected attention architecture, whose comparison and content
        maps have the same :math:`4K^2` parameters per layer as dense attention. Historical mappings
        with head-scoped interfaces require explicit architecture migration. A base with
        ``tie_comparison_maps=True`` (:math:`K^2` fewer comparison parameters per layer) or
        ``prior_covariance_table='block_cholesky'`` (another per-type table count) is refused. A base with
        ``learn_attention_temperature=True`` is accepted: its :math:`HL` routing log-temperatures are
        context-processing parameters the matched arm does not have (``paper_arms`` notes the count).
        """
        values = _vfe_base_values(base)
        for field in ("relational_input_scope", "relational_output_scope"):
            if field in values and values[field] not in (None, "full"):
                raise ValueError(f"matched_to: {field}={values[field]!r}: head architecture requires explicit migration")
        if values["attention_mode"] not in ("projected_predictive", "projected_gradient"):
            raise ValueError(f"matched_to: retired attention mode {values['attention_mode']!r} requires explicit migration")
        # Dense equivalence (audit 2026-10-07 F79): the minimal-agent readout, content-link edges and a
        # missing block MLP each change the context-processing parameter count, so the matched arm
        # would silently differ from the base in size.
        if values["predictive_readout"] != "kernel":
            raise ValueError(f"matched_to: predictive_readout={values['predictive_readout']!r} is not "
                             "dense-equivalent; only the kernel readout has the dense attention parameter count")
        if values["content_link_edges"] != "off":
            raise ValueError(f"matched_to: content_link_edges={values['content_link_edges']!r} adds edge-link "
                             "parameters that dense attention does not have")
        if not values["use_block_mlp"]:
            raise ValueError("matched_to: the base has no block MLP, while the matched arm always builds one")
        # Audit 2026-10-09 round 2 item 19: the tie drops the key comparison map (K^2 per layer fewer
        # context-processing parameters than dense attention) and the frame-free table replaces the
        # (V, K) log-variance and V K d frame tables by (V, H, d(d+1)/2) Cholesky entries, so either one
        # breaks the parameter equivalence the matched arm rests on.
        if values.get("tie_comparison_maps"):
            raise ValueError("matched_to: tie_comparison_maps=True shares S_K = S_Q, so the base has K^2 fewer "
                             "comparison parameters per layer than dense attention (W_Q, W_K); the matched "
                             "context-processing count does not hold")
        if values.get("prior_covariance_table", "diagonal") != "diagonal":
            raise ValueError(f"matched_to: prior_covariance_table={values['prior_covariance_table']!r} changes the "
                             "per-type table count (no frame table; head-block Cholesky entries instead of "
                             "diagonal log variances), which the matched per-type read map is sized against")
        # The per-type map is V*K*d, the frame-table size of untied block_glk only (F81).
        if per_type_table != "none" and (values["gauge_group"] != "block_glk"
                                         or values["gauge_tying"] not in (None, "per_head")):
            raise ValueError(f"matched_to: per_type_table={per_type_table!r} is sized V*K*d, which equals the "
                             "base's frame table only for gauge_group='block_glk' with gauge_tying='per_head'; got "
                             f"{values['gauge_group']!r}/{values['gauge_tying']!r}")
        derived: Dict[str, Any] = {name: values[name] for name in MATCHED_FIELDS}
        derived["mlp_expansion"] = int(values["block_mlp_expansion"])
        derived["decode_bias"] = True                    # the VFE linear decoder always carries the bias
        # rope_mode, or the pos_rotation of a base saved before the 2026-10-12 merge.
        rope_active = values.get("rope_mode", "none") != "none" or values.get("pos_rotation") == "rope"
        derived["pos_rotation"] = "rope" if rope_active else "none"
        derived["rope_base"] = float(values.get("rope_base", 100.0))
        derived["rope_fraction"] = float(values.get("rope_fraction", 1.0))
        derived["per_type_table"] = per_type_table
        phi_scale = values.get("phi_scale")
        if isinstance(phi_scale, (int, float)) and not isinstance(phi_scale, bool) and phi_scale > 0:
            derived["per_type_table_init_scale"] = float(phi_scale)
        derived.update(overrides)
        if derived.get("norm_placement") == "post_shared" and "attention_residual" not in overrides:
            derived["attention_residual"] = bool(values["attention_residual"])
        derived["model_family"] = MODEL_FAMILY
        return cls(**derived)

    def with_width(self, embed_dim: int, *, head_dim: Optional[int] = None) -> "BaselineConfig":
        r"""The same config at width ``embed_dim`` with heads of ``head_dim`` (default: the current :math:`d`)."""
        d = self.d_head if head_dim is None else int(head_dim)
        if embed_dim % d:
            raise ValueError(f"width {embed_dim} is not a multiple of the head width {d}")
        return replace(self, embed_dim=int(embed_dim), n_heads=int(embed_dim) // d)


def _vfe_base_values(base: Any) -> Dict[str, Any]:
    r"""Field values of a VFE base (config object or mapping), current VFE3Config defaults for gaps."""
    from dataclasses import MISSING

    from vfe3.config import VFE3Config, _normalize_serialized_boolean

    if is_dataclass(base) and not isinstance(base, type):
        if model_family_of(base) != VFE_MODEL_FAMILY:
            raise ValueError(f"matched_to needs a VFE base config, got model_family={model_family_of(base)!r}")
        raw = {f.name: getattr(base, f.name) for f in fields(base)}
    elif isinstance(base, Mapping):
        inner = base.get("config")
        raw = dict(inner) if isinstance(inner, Mapping) else dict(base)
        if model_family_of(raw) != VFE_MODEL_FAMILY:
            raise ValueError(f"matched_to needs a VFE base config, got model_family={model_family_of(raw)!r}")
    else:
        raise TypeError(f"matched_to expects a VFE3Config or a mapping, got {type(base).__name__}")
    if "relational_mode" in raw or "relational_readout_mode" in raw:
        pair = (raw.pop("relational_mode", "off"),
                raw.pop("relational_readout_mode", "conditional_update"))
        if pair != ("projected_filtering", "predictive_content"):
            raise ValueError(f"matched_to: retired attention selectors {pair!r} require explicit migration")
        if "attention_mode" in raw and raw["attention_mode"] != "projected_predictive":
            raise ValueError("matched_to: conflicting attention selectors")
        raw["attention_mode"] = "projected_predictive"
    if "relational_identity_content" in raw:
        identity = _normalize_serialized_boolean(
            raw.pop("relational_identity_content"), "relational_identity_content", source="matched_to")
        if identity:
            raise ValueError("matched_to: retired identity content requires explicit migration")
    if "attention_mode" not in raw:
        raise ValueError("matched_to: base config names no attention selector; it predates the "
                         "projected routes and requires explicit migration")
    defaults: Dict[str, Any] = {}
    for f in fields(VFE3Config):
        if f.default is not MISSING:
            defaults[f.name] = f.default
        elif f.default_factory is not MISSING:                          # type: ignore[misc]
            defaults[f.name] = f.default_factory()                      # type: ignore[misc]
    return {**defaults, **raw}


def baseline_config_from_serialized(
    payload: Mapping[str, Any],

    *,
    source: str,
    drop_unknown_config_fields: bool = False,
) -> Tuple[BaselineConfig, Dict[str, Any]]:
    r"""``(BaselineConfig, dropped)``; with ``drop_unknown_config_fields`` undeclared fields are dropped."""
    dropped: Dict[str, Any] = {}
    if drop_unknown_config_fields and isinstance(payload, Mapping):
        known = {f.name for f in fields(BaselineConfig)}
        dropped = {str(k): v for k, v in payload.items() if k not in known}
        payload = {k: v for k, v in payload.items() if k in known}
    return BaselineConfig.from_serialized(payload, source=source), dropped


# ----------------------------------------------------------------------------------------------
# Parameter roles
# ----------------------------------------------------------------------------------------------

# run_resources role -> vfe3.train named parameter group (PARAMETER_GROUPS).
_TRAIN_GROUP_BY_ROLE: Dict[str, str] = {
    "token_embed":        "mu_embed",
    "type_table":         "per_type_table",
    "decode":             "decode",
    "relational_score":   "relational_score",
    "relational_content": "relational_content",
    "block_mlp":          "block_mlp",
    "norm":               "norm",
}


def _canonical(name: str) -> str:
    stripped = True
    while stripped:
        stripped = False
        for prefix in _WRAPPER_PREFIXES:
            if name.startswith(prefix):
                name, stripped = name[len(prefix):], True
    return name


def baseline_parameter_role(name: str) -> Tuple[str, str]:
    r"""``(section, role)`` of a :class:`BaselineTransformer` parameter, in the sections of
    :mod:`vfe3.run_resources`.

    ``per_type_tables``: ``tok_embed`` (role ``token_embed``), ``type_table`` (``type_table``) and the
    decoder ``W_out``, ``b_out`` (``decode``). ``context_processing``: ``W_Q``, ``W_K``
    (``relational_score``, the comparison maps), ``W_V``, ``W_O`` (``relational_content``), the MLP
    (``block_mlp``) and every normalization (``norm``). The role names are the VFE's, so the Table M2
    context/total cells compare like with like.
    """
    name = _canonical(name)
    leaf = name.rsplit(".", 1)[-1]
    if name == "tok_embed":
        return "per_type_tables", "token_embed"
    if name == "type_table":
        return "per_type_tables", "type_table"
    if name.startswith("decoder.") and leaf in ("output_proj_weight", "output_proj_bias"):
        return "per_type_tables", "decode"
    if name.startswith("blocks."):
        if ".attn." in name and leaf in ("w_q", "w_k"):
            return "context_processing", "relational_score"
        if ".attn." in name and leaf in ("w_v", "w_o"):
            return "context_processing", "relational_content"
        if ".mlp." in name:
            return "context_processing", "block_mlp"
        if ".ln1." in name or ".ln2." in name:
            return "context_processing", "norm"
    if name.startswith(("final_norm.", "shared_norm.")):
        return "context_processing", "norm"
    return "other", leaf


def baseline_parameter_group(name: str) -> Optional[str]:
    r"""The :data:`vfe3.train.PARAMETER_GROUPS` entry of a baseline parameter, or None when unknown."""
    section, role = baseline_parameter_role(name)
    return None if section == "other" else _TRAIN_GROUP_BY_ROLE.get(role)


# ----------------------------------------------------------------------------------------------
# Modules
# ----------------------------------------------------------------------------------------------

class LinearUnigramDecoder(nn.Module):
    r"""The VFE linear decoder, :math:`W_{\rm out} h + b_{\rm out} + \kappa \log\pi`, on the baseline.

    It duck-types the :class:`~vfe3.model.prior_bank.PriorBank` attributes of the linear decode, so the
    training CE is :meth:`PriorBank.decode_ce_linear_chunked` itself (the identical fused chunked fp32
    kernel) and the unigram table is filled by :meth:`PriorBank.set_unigram_log_prior` (the identical
    add-one formula). ``W_out`` is Xavier-uniform and ``b_out`` starts at zero, as in the VFE.
    """

    # The PriorBank linear-decode kernels, bound to this module's identically named attributes.
    _validate_fused_ce_targets = PriorBank._validate_fused_ce_targets
    _unigram_bias = PriorBank._unigram_bias
    set_unigram_log_prior = PriorBank.set_unigram_log_prior
    decode_ce_linear_chunked = PriorBank.decode_ce_linear_chunked

    def __init__(
        self,
        vocab_size: int,
        embed_dim:  int,

        *,
        bias:          bool,
        unigram_prior: bool,
        unigram_kappa: float,
        chunk_size:    int,
        checkpoint:    str,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.decode_chunk_size = int(chunk_size)
        self.decode_ce_checkpoint = str(checkpoint)
        # The shared kernel pulls the mean back through a frame only under 'reference'; the baseline
        # has no frames, so its decoder always reads the hidden state in place.
        self.decode_coordinates = "local"
        self.decode_unigram_prior = bool(unigram_prior)
        self.unigram_kappa = float(unigram_kappa)
        self.output_proj_weight = nn.Parameter(torch.empty(self.vocab_size, int(embed_dim)))  # (V, K)
        nn.init.xavier_uniform_(self.output_proj_weight)
        if bias:
            self.output_proj_bias = nn.Parameter(torch.zeros(self.vocab_size))              # (V,)
        else:
            self.register_parameter("output_proj_bias", None)
        if self.decode_unigram_prior:
            self.register_buffer("unigram_log_prior", torch.zeros(self.vocab_size))        # (V,) log pi
            self._unigram_set = False

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        r"""Dense fp32 logits ``(B, N, V)`` from the final hidden state ``h`` ``(B, N, K)``."""
        out = h.float() @ self.output_proj_weight.float().transpose(-1, -2)               # (B, N, V)
        if self.output_proj_bias is not None:
            out = out + self.output_proj_bias.float()
        if self.decode_unigram_prior:
            out = out + self._unigram_bias().float()
        return out


class BaselineAttention(nn.Module):
    r"""Bias-free multi-head attention with an additive per-head log-prior and an explicit softmax."""

    def __init__(self, embed_dim: int, n_heads: int, init_std: float, out_std: float) -> None:
        super().__init__()
        self.embed_dim, self.n_heads = int(embed_dim), int(n_heads)
        self.d_head = self.embed_dim // self.n_heads
        K = self.embed_dim
        self.w_q = nn.Parameter(torch.empty(K, K))                                          # (K_out, K_in)
        self.w_k = nn.Parameter(torch.empty(K, K))
        self.w_v = nn.Parameter(torch.empty(K, K))
        self.w_o = nn.Parameter(torch.empty(K, K))
        for weight in (self.w_q, self.w_k, self.w_v):
            nn.init.normal_(weight, 0.0, init_std)
        nn.init.normal_(self.w_o, 0.0, out_std)

    def forward(
        self,
        u:     torch.Tensor,                 # (B, N, K) read attention input
        prior: torch.Tensor,                 # (H, N, N) fp32 log-prior (-inf off the causal support)
        rope:  Optional[torch.Tensor] = None,  # (N, d, d) per-head rotary map R_i, or None

        *,
        weights_out: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:                       # (B, N, K)
        B, N, K = u.shape
        H, d = self.n_heads, self.d_head
        q = (u @ self.w_q.transpose(0, 1)).view(B, N, H, d).transpose(1, 2)                # (B, H, N, d)
        k = (u @ self.w_k.transpose(0, 1)).view(B, N, H, d).transpose(1, 2)                # (B, H, N, d)
        if rope is not None:
            # q_i -> R_i q_i, k_j -> R_j k_j, so q_i^T k_j -> q_i^T R_{j-i} k_j.
            q = torch.einsum("nst,bhnt->bhns", rope.to(q.dtype), q)                        # (B, H, N, d)
            k = torch.einsum("nst,bhnt->bhns", rope.to(k.dtype), k)                        # (B, H, N, d)
        v = (u @ self.w_v.transpose(0, 1)).view(B, N, H, d).transpose(1, 2)                # (B, H, N, d)
        scores = (q @ k.transpose(-1, -2)).float() * (1.0 / math.sqrt(d)) + prior          # (B, H, N, N) fp32
        beta = torch.softmax(scores, dim=-1)                                                # (B, H, N, N) fp32
        if weights_out is not None:
            weights_out.append(beta.detach())
        z = (beta.to(v.dtype) @ v).transpose(1, 2).reshape(B, N, K)                         # (B, N, K)
        return z @ self.w_o.transpose(0, 1)                                                 # (B, N, K)


class BaselineMLP(nn.Module):
    r"""``W_2 GELU_erf(W_1 u + b_1) + b_2`` of hidden width ``e K``."""

    def __init__(self, embed_dim: int, expansion: int, init_std: float, out_std: float) -> None:
        super().__init__()
        hidden = int(expansion) * int(embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.fc2 = nn.Linear(hidden, embed_dim)
        nn.init.normal_(self.fc1.weight, 0.0, init_std)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, 0.0, out_std)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, u: torch.Tensor) -> torch.Tensor:                                    # (B, N, K)
        return self.fc2(F.gelu(self.fc1(u), approximate="none"))


class BaselineBlock(nn.Module):
    r"""One layer: attention on the (optionally read-mapped) state, then the MLP, pre-LN or post-shared.

    ``read`` and ``write`` are per-token head-block maps gathered once per forward. Under
    ``per_type_table='block_read'`` ``read`` holds :math:`X_{x_i}` (applied as :math:`I + X`) and
    ``write`` is None; under ``'block_rw'`` they hold :math:`\exp(-X_{x_i})` and :math:`\exp(X_{x_i})`.
    Under ``norm_placement='post_shared'`` the block owns no normalization and applies ``shared_norm``.
    """

    def __init__(self, cfg: BaselineConfig) -> None:
        super().__init__()
        K = cfg.embed_dim
        out_std = cfg.init_std / math.sqrt(2.0 * cfg.n_layers)
        self.pre_ln = cfg.norm_placement == "pre"
        self.read_is_increment = cfg.per_type_table == "block_read"
        self.attention_residual = cfg.attention_residual
        if self.pre_ln:
            self.ln1 = nn.LayerNorm(K, eps=cfg.layernorm_eps)
        self.attn = BaselineAttention(K, cfg.n_heads, cfg.init_std, out_std)
        if self.pre_ln:
            self.ln2 = nn.LayerNorm(K, eps=cfg.layernorm_eps)
        self.mlp = BaselineMLP(K, cfg.mlp_expansion, cfg.init_std, out_std)

    @staticmethod
    def _apply_blocks(maps: torch.Tensor, x: torch.Tensor, increment: bool) -> torch.Tensor:
        B, N, K = x.shape
        x_h = x.view(B, N, maps.shape[2], maps.shape[3])                                    # (B, N, H, d)
        y = torch.einsum("bnhij,bnhj->bnhi", maps.to(x_h.dtype), x_h)                     # (B, N, H, d)
        # block_read: A u = u + X u, exactly u when X = 0; block_rw: the map is the whole block.
        return (x_h + y if increment else y).reshape(B, N, K)                               # (B, N, K)

    def forward(
        self,
        h:     torch.Tensor,                 # (B, N, K) residual stream (fp32)
        prior: torch.Tensor,                 # (H, N, N) fp32 log-prior
        read:  Optional[torch.Tensor],       # (B, N, H, d, d) gathered read map, or None
        write: Optional[torch.Tensor] = None,  # (B, N, H, d, d) gathered write map, or None

        *,
        weights_out: Optional[List[torch.Tensor]] = None,
        shared_norm: Optional[nn.LayerNorm] = None,
        rope:        Optional[torch.Tensor] = None,  # (N, d, d) per-head rotary map, or None
    ) -> torch.Tensor:                       # (B, N, K)
        u = _fp32_layer_norm(self.ln1, h) if self.pre_ln else h                             # (B, N, K)
        if read is not None:
            u = self._apply_blocks(read, u, self.read_is_increment)                         # (B, N, K)
        a = self.attn(u, prior, rope, weights_out=weights_out)                              # (B, N, K)
        if write is not None:
            a = self._apply_blocks(write, a, False)                                         # (B, N, K)
        if self.pre_ln:
            h = h + a
            return h + self.mlp(_fp32_layer_norm(self.ln2, h))
        if shared_norm is None:
            raise ValueError("norm_placement='post_shared' needs the model's shared_norm")
        h = _fp32_layer_norm(shared_norm, h + a if self.attention_residual else a)          # (B, N, K) fp32
        return h + self.mlp(h)


def _fp32_layer_norm(norm: nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
    r"""LayerNorm evaluated in fp32 whatever the ambient autocast (``(..., K)`` in, fp32 out)."""
    with torch.autocast(device_type=x.device.type, enabled=False):
        return F.layer_norm(x.float(), norm.normalized_shape, norm.weight.float(), norm.bias.float(),
                            norm.eps)


class BaselineTransformer(nn.Module):
    r"""Pre-LN GPT decoder with the VFE's attention prior and linear decoder (see the module docstring).

    Implements the interface of the shared loop: ``forward(token_ids, targets=None, *, estep_grad_out,
    return_decode_stats)``, ``forward_beliefs``, ``diagnostics``, ``attention_maps``, ``generate``,
    ``set_unigram_log_prior``, ``build_optimizer``, ``cost_model_fields`` and
    ``_attention_log_prior``. ``model_family`` marks it for :func:`vfe3.contracts.is_vfe_model`.
    """

    model_family: str = MODEL_FAMILY

    def __init__(self, cfg: BaselineConfig) -> None:
        super().__init__()
        if not isinstance(cfg, BaselineConfig):
            raise TypeError(f"BaselineTransformer needs a BaselineConfig, got {type(cfg).__name__}")
        from vfe3.runtime import set_fill_uninitialized_memory, set_fp32_matmul_precision

        set_fp32_matmul_precision(cfg.fp32_matmul_precision)
        set_fill_uninitialized_memory(cfg.fill_uninitialized_memory)
        self.cfg = cfg
        V, K, H, d = cfg.vocab_size, cfg.embed_dim, cfg.n_heads, cfg.d_head
        self.tok_embed = nn.Parameter(torch.empty(V, K))                                   # (V, K) E
        nn.init.normal_(self.tok_embed, 0.0, cfg.init_std)
        self.blocks = nn.ModuleList([BaselineBlock(cfg) for _ in range(cfg.n_layers)])
        if cfg.norm_placement == "pre":
            self.final_norm = nn.LayerNorm(K, eps=cfg.layernorm_eps)
            self.shared_norm = None
        else:
            self.final_norm = None
            self.shared_norm = nn.LayerNorm(K, eps=cfg.layernorm_eps)
        self.decoder = LinearUnigramDecoder(
            V, K, bias=cfg.decode_bias, unigram_prior=cfg.decode_unigram_prior,
            unigram_kappa=cfg.unigram_kappa, chunk_size=cfg.decode_chunk_size,
            checkpoint=cfg.decode_ce_checkpoint)
        # The read map draws from its own generator (or none at zero init), so every other
        # parameter is initialized identically with and without it at the same seed.
        if cfg.per_type_table != "none":
            table = torch.zeros(V, H, d, d)                                                 # (V, H, d, d) X
            if cfg.per_type_table_init == "phi_like" and cfg.per_type_table_init_scale > 0.0:
                generator = torch.Generator().manual_seed(_derived_seed(cfg.seed, "type_table"))
                table = torch.randn(V, H, d, d, generator=generator) * cfg.per_type_table_init_scale
            self.type_table = nn.Parameter(table)
        else:
            self.register_parameter("type_table", None)
        self._log_prior_cache: Dict[Tuple[Any, ...], torch.Tensor] = {}

    # -- precision -----------------------------------------------------------------------------

    def _amp_context(self, device: torch.device) -> contextlib.AbstractContextManager:
        if self.cfg.amp_dtype is None:
            return contextlib.nullcontext()
        dtype = torch.bfloat16 if self.cfg.amp_dtype == "bf16" else torch.float16
        return torch.autocast(device_type=device.type, dtype=dtype)

    def _amp_off_context(self, device: torch.device) -> contextlib.AbstractContextManager:
        if self.cfg.amp_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=device.type, enabled=False)

    # -- core ----------------------------------------------------------------------------------

    def _attention_log_prior(self, n: int, device: torch.device) -> torch.Tensor:
        r"""The ``(H, N, N)`` fp32 attention log-prior, the VFE registry's (cached on ``(N, device)``)."""
        from vfe3.attention_prior import attention_log_prior

        key = (self.cfg.beta_attention_prior, int(n), str(device), self.cfg.n_heads, self.cfg.alibi_slope)
        cached = self._log_prior_cache.get(key)
        if cached is None:
            cached = attention_log_prior(
                self.cfg.beta_attention_prior, int(n), int(n), device=device, dtype=torch.float32,
                n_heads=self.cfg.n_heads, alibi_slope=self.cfg.alibi_slope)
            if cached.dim() == 2:                                                           # (N, N) head-free
                cached = cached.unsqueeze(0).expand(self.cfg.n_heads, -1, -1)
            self._log_prior_cache[key] = cached
        return cached

    def _rope_rotation(self, n: int, device: torch.device) -> Optional[torch.Tensor]:
        r"""The ``(N, d, d)`` fp32 per-head rotary map :math:`R_i` under ``pos_rotation='rope'``, else None."""
        if self.cfg.pos_rotation != "rope":
            return None
        from vfe3.geometry.rope import build_rope_rotation

        key = ("rope", self.cfg.rope_base, self.cfg.rope_fraction, int(n), str(device), self.cfg.d_head)
        cached = self._log_prior_cache.get(key)
        if cached is None:
            cached = build_rope_rotation(torch.arange(int(n), device=device), [self.cfg.d_head],
                                         base=self.cfg.rope_base, fraction=self.cfg.rope_fraction,
                                         device=device, dtype=torch.float32)
            self._log_prior_cache[key] = cached
        return cached

    def hidden(
        self,
        token_ids: torch.Tensor,             # (B, N) token ids

        *,
        weights_out: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:                       # (B, N, K) fp32 final-normalized state
        r"""The final normalized hidden state :math:`\mathrm{LN}_f(h^{(L)})` (fp32)."""
        if token_ids.dim() != 2:
            raise ValueError(f"token_ids must be (B, N), got {tuple(token_ids.shape)}")
        device = token_ids.device
        prior = self._attention_log_prior(token_ids.shape[1], device)                      # (H, N, N)
        rope = self._rope_rotation(token_ids.shape[1], device)                              # (N, d, d) or None
        h = self.tok_embed[token_ids]                                                       # (B, N, K)
        read, write = None, None
        if self.type_table is not None:
            X = self.type_table[token_ids]                                                  # (B, N, H, d, d)
            if self.cfg.per_type_table == "block_rw":
                with torch.autocast(device_type=device.type, enabled=False):
                    X = X.float()
                    read, write = torch.linalg.matrix_exp(-X), torch.linalg.matrix_exp(X)  # (B, N, H, d, d)
            else:
                read = X
        with self._amp_context(device):
            for block in self.blocks:
                h = block(h, prior, read, write, weights_out=weights_out, shared_norm=self.shared_norm,
                          rope=rope)
        if self.final_norm is None:
            return h.float()                                                                # (B, N, K) fp32
        return _fp32_layer_norm(self.final_norm, h)                                         # (B, N, K) fp32

    def forward(
        self,
        token_ids: torch.Tensor,                          # (B, N)
        targets:   Optional[torch.Tensor] = None,         # (B, N) next-token ids (-100 = ignore)

        *,
        estep_grad_out: Optional[Dict[str, float]] = None,
        return_decode_stats: bool = False,
    ) -> Any:
        r"""Logits ``(B, N, V)`` without targets; ``(None, loss, ce | DecodeCEResult)`` with targets.

        The CE is the fused chunked mean over the targets that are not ``-100``,
        :math:`\mathrm{CE} = \frac{1}{|S|}\sum_{i \in S} -\log p(y_i \mid y_{<i})`. There is no E-step,
        so an ``estep_grad_out`` dict is filled with NaN (not the loop's 0.0 placeholder).
        """
        if estep_grad_out is not None:
            for key in ("mu", "sigma", "phi"):
                estep_grad_out[key] = float("nan")
        h = self.hidden(token_ids)                                                          # (B, N, K)
        if targets is None:
            with self._amp_off_context(token_ids.device):
                return self.decoder.logits(h)                                               # (B, N, V)
        with self._amp_off_context(token_ids.device):
            result = self.decoder.decode_ce_linear_chunked(h.float(), targets, return_stats=True)
        ce = result.ce
        loss = ce
        if return_decode_stats:
            return None, loss, DecodeCEResult(ce.detach(), result.scored_tokens, result.excluded_tokens)
        return None, loss, ce.detach()

    def forward_beliefs(
        self,
        token_ids: torch.Tensor,             # (B, N)

        *,
        return_logits: bool = False,
        capture:       Optional[dict] = None,
        estep_grad_out: Optional[Dict[str, float]] = None,
        **_unused: Any,
    ) -> Tuple[None, Optional[torch.Tensor]]:
        r"""``(None, logits or None)``: the VFE inference seam's shape. There is no belief state."""
        if estep_grad_out is not None:
            for key in ("mu", "sigma", "phi"):
                estep_grad_out[key] = float("nan")
        h = self.hidden(token_ids)
        if not return_logits:
            return None, None
        with self._amp_off_context(token_ids.device):
            return None, self.decoder.logits(h)

    def set_unigram_log_prior(self, counts: torch.Tensor) -> None:
        r"""Fill :math:`\log\pi_v = \log((c_v + 1)/(\sum_u c_u + V))` from training counts ``(V,)``."""
        self.decoder.set_unigram_log_prior(counts)

    @torch.no_grad()
    def attention_maps(
        self,
        token_ids: torch.Tensor,             # (B, N); the first sequence is used

        *,
        score_at:  str = "entry",
    ) -> torch.Tensor:                       # (L, H, N, N) attention weights of sequence 0
        r"""The attention weights the forward used, per layer and head, on the first sequence."""
        weights: List[torch.Tensor] = []
        self.hidden(token_ids[:1], weights_out=weights)
        return torch.stack([w[0].float() for w in weights])

    @torch.no_grad()
    def diagnostics(self, token_ids: torch.Tensor) -> Dict[str, float]:
        r"""The keys the shared loop logs. ``attn_entropy`` is the mean row entropy of the attention
        weights of sequence 0 over layers and heads; the VFE free-energy and geometry terms are NaN."""
        from vfe3.metrics import attention_entropy

        maps = self.attention_maps(token_ids)                                               # (L, H, N, N)
        nan = float("nan")
        return {
            "attn_entropy": float(attention_entropy(maps)),
            "self_coupling": nan, "self_divergence": nan, "belief_coupling": nan,
            "attention_entropy": nan, "total": nan, "effective_rank": nan,
            "holonomy_deviation": nan, "gauge_trace_spread": nan,
        }

    @torch.no_grad()
    def generate(
        self,
        token_ids:      torch.Tensor,        # (B, N0) prompt
        max_new_tokens: int,

        *,
        temperature: float = 1.0,
        greedy:      bool = False,
        top_k:       Optional[int] = None,
    ) -> torch.Tensor:                       # (B, N0 + max_new_tokens)
        r"""Autoregressive continuation over the last ``max_seq_len`` tokens (greedy or sampled)."""
        if max_new_tokens < 0:
            raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")
        if not greedy and not temperature > 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        seq = token_ids
        for _ in range(int(max_new_tokens)):
            logits = self.forward(seq[:, -self.cfg.max_seq_len:])[:, -1, :]               # (B, V)
            if greedy:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, int(top_k), dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1)
            seq = torch.cat([seq, nxt], dim=1)
        return seq

    # -- optimization and accounting -----------------------------------------------------------

    def build_optimizer(self, cfg: Optional[BaselineConfig] = None) -> torch.optim.Optimizer:
        r"""See :func:`build_optimizer`."""
        return build_optimizer(self, cfg if cfg is not None else self.cfg)

    def context_processing_parameter_count(self) -> int:
        r"""Measured context-processing parameters (the ``context_processing`` role section)."""
        return sum(p.numel() for name, p in self.named_parameters()
                   if baseline_parameter_role(name)[0] == "context_processing")

    def cost_model_fields(
        self,
        n_params:    int,
        tokens_seen: int,

        *,
        wall_time:   Optional[float] = None,
    ) -> Dict[str, object]:
        r"""Structural axes and an analytic FLOP count (Kaplan et al. 2020, arXiv:2001.08361, Table 1).

        Forward FLOPs per token :math:`C_f = 2(N_{\rm mm} + VK) + 2 L N_{\rm ctx} K`, with
        :math:`N_{\rm mm} = L(4K^2 + 2eK^2) + L K d\,[\text{read map}]` the matrix-multiply
        parameters and :math:`2 L N_{\rm ctx} K` the attention scores and weighted values; training is
        :math:`3 C_f` per token. An analytic proxy, not a measurement: the matched-compute arm uses
        measured device-hours.
        """
        from vfe3.run_artifacts import _execution_device_fields

        cfg = self.cfg
        K, L, N, V, e, d = (cfg.embed_dim, cfg.n_layers, cfg.max_seq_len, cfg.vocab_size,
                            cfg.mlp_expansion, cfg.d_head)
        table_mm = {"none": 0, "block_read": L * K * d, "block_rw": 2 * L * K * d}[cfg.per_type_table]
        matmul = L * (4 * K * K + 2 * e * K * K) + table_mm
        forward_per_token = 2.0 * (matmul + V * K) + 2.0 * L * N * K
        out: Dict[str, object] = {
            "model_family":            MODEL_FAMILY,
            "embed_dim":               K,
            "n_heads":                 int(cfg.n_heads),
            "d_head":                  d,
            "n_layers":                L,
            "mlp_expansion":           e,
            "max_seq_len":             N,
            "batch_size":              int(cfg.batch_size),
            "vocab_size":              V,
            "decode_mode":             cfg.decode_mode,
            "per_type_table":          cfg.per_type_table,
            "n_learnable_params":      int(sum(p.numel() for p in self.parameters() if p.requires_grad)),
            "n_context_processing_params": int(self.context_processing_parameter_count()),
            "flops_per_token_forward": forward_per_token,
            "flops_per_token_train":   3.0 * forward_per_token,
            "est_flops_analytic":      3.0 * forward_per_token * float(tokens_seen),
            **_execution_device_fields(self),
            "amp_dtype":               cfg.amp_dtype,
        }
        if wall_time is not None and tokens_seen > 0:
            out["wall_time_s"] = float(wall_time)
            out["wall_time_per_token"] = float(wall_time) / float(tokens_seen)
            out["wall_time_per_step"] = float(wall_time) / max(1, int(tokens_seen) // (int(cfg.batch_size)
                                                                                     * int(cfg.max_seq_len)))
        return out


def build_optimizer(
    model: BaselineTransformer,
    cfg:   BaselineConfig,
) -> torch.optim.Optimizer:
    r"""AdamW over three groups, each with a role tag the shared loop reads.

    Matrices and tables (:math:`E`, :math:`W_{Q,K,V,O}`, :math:`W_{1,2}`, :math:`W_{\rm out}`) decay
    at ``weight_decay``; gains and biases (LayerNorm, :math:`b_{1,2}`, :math:`b_{\rm out}`) do not.
    Both carry role ``"mu"``, so under ``grad_clip_scope='role'`` the loop's per-role clip is one global
    clip over the transformer; under ``'group'`` each group below is clipped by its own norm.
    The read-map table :math:`X` has its own group (``per_type_table_lr``, ``per_type_table_weight_decay``,
    decaying :math:`A` toward :math:`I`) with role ``"phi"``: like the VFE frame table it replaces, it
    gets its own clip budget and the ``lr_phi`` / ``grad_norm_phi`` columns. Every trainable parameter
    must land in exactly one group. With ``hidden_lr_scale`` :math:`s \ne 1` the hidden matrices
    (:math:`W_{Q,K,V,O}`, :math:`W_{1,2}`) form a fourth group at :math:`s \cdot` ``lr`` with the same decay
    and role (the muP Adam rule for hidden weights, Yang et al. 2022, arXiv:2203.03466, Table 3); :math:`E`
    and :math:`W_{\rm out}` keep ``lr``. At :math:`s = 1` the groups are the three above.
    """
    decay: List[nn.Parameter] = []
    hidden: List[nn.Parameter] = []
    no_decay: List[nn.Parameter] = []
    table: List[nn.Parameter] = []
    split_hidden = cfg.hidden_lr_scale != 1.0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if _canonical(name) == "type_table":
            table.append(parameter)
        elif parameter.dim() >= 2:
            is_hidden = _canonical(name).startswith("blocks.")
            (hidden if split_hidden and is_hidden else decay).append(parameter)
        else:
            no_decay.append(parameter)
    groups: List[Dict[str, Any]] = [
        {"params": decay, "lr": cfg.lr, "weight_decay": cfg.weight_decay, "role": "mu",
         "lr_report_role": "mu"},
        {"params": no_decay, "lr": cfg.lr, "weight_decay": 0.0, "role": "mu"},
    ]
    if hidden:
        groups.append({"params": hidden, "lr": cfg.lr * cfg.hidden_lr_scale, "weight_decay": cfg.weight_decay,
                       "role": "mu"})
    if table:
        groups.append({
            "params": table,
            "lr": cfg.lr if cfg.per_type_table_lr is None else cfg.per_type_table_lr,
            "weight_decay": (cfg.weight_decay if cfg.per_type_table_weight_decay is None
                             else cfg.per_type_table_weight_decay),
            "role": "phi", "lr_report_role": "phi",
        })
    groups = [group for group in groups if group["params"]]
    grouped = {id(p) for group in groups for p in group["params"]}
    missing = [name for name, p in model.named_parameters() if p.requires_grad and id(p) not in grouped]
    if missing:
        raise AssertionError(f"baseline build_optimizer left parameters ungrouped: {missing}")
    fused = next(model.parameters()).is_cuda
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.adam_beta1, cfg.adam_beta2),
                             eps=cfg.adam_eps, weight_decay=cfg.weight_decay, fused=fused)


# ----------------------------------------------------------------------------------------------
# Compute matching (Table M2, "matched measured compute")
# ----------------------------------------------------------------------------------------------

def steps_for_device_hours(
    tokens_per_s:    float,
    hours:           float,
    tokens_per_step: int,
) -> int:
    r"""Optimizer steps that spend ``hours`` device-hours at a measured training throughput.

    .. math:: S = \Big\lfloor \frac{3600\, h\, \tau}{T_{\rm step}} \Big\rfloor,

    with :math:`\tau` the measured training tokens per second and :math:`T_{\rm step}` the tokens per
    optimizer step (batch times context).
    """
    for name, value in (("tokens_per_s", tokens_per_s), ("hours", hours)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
                or value <= 0:
            raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    if type(tokens_per_step) is not int or tokens_per_step < 1:
        raise ValueError(f"tokens_per_step must be a positive int, got {tokens_per_step!r}")
    return int(math.floor(3600.0 * float(hours) * float(tokens_per_s) / tokens_per_step))


WIDTH_LADDER_RULE: str = (
    "Fixed budget: the measured training device-hours h of the retrained evaluated arm (same code, "
    "same GPU, idle device). For every width d on the ladder a short calibration run gives the "
    "training throughput tau(d); the tokens the budget buys are T(d) = 3600 h tau(d). The required "
    "token count is D(d) = min(20 P_ctx(d), 16 U), 20 context-processing parameters' worth of tokens "
    "(Hoffmann et al. 2022) capped at 16 passes over the U unique training tokens (Muennighoff et al. "
    "2023). The chosen width is the largest d with T(d) >= D(d) (the smallest ladder width, flagged, "
    "when none qualifies). The whole budget is then spent (max_steps = floor(T(d) / tokens_per_step), "
    "tokens free), depth, head width, batch, context and schedule shape stay those of the matched arm, "
    "and the validation-selected checkpoint is reported. The cap enters the width choice only.")


def width_ladder_rule(
    base: BaselineConfig,

    *,
    hours:                float,
    tokens_per_s:         Mapping[int, float],   # measured training tokens/s per ladder width
    unique_train_tokens:  int,
    tokens_per_parameter: float = TOKENS_PER_CONTEXT_PARAMETER,
    max_passes:           float = MAX_PASSES,
    head_dim:             Optional[int] = None,
) -> Dict[str, Any]:
    r"""The pre-declared width rule of the compute-matched arm (:data:`WIDTH_LADDER_RULE`).

    For each ladder width :math:`d` (the keys of ``tokens_per_s``), with :math:`U` unique training
    tokens and :math:`h` device-hours,

    .. math::
        T(d) = 3600\, h\, \tau(d),\qquad
        D(d) = \min\big(c\, P_{\rm ctx}(d),\; p_{\max} U\big),\qquad
        d^\star = \max\{d : T(d) \ge D(d)\},

    with :math:`c` = ``tokens_per_parameter`` (20) and :math:`p_{\max}` = ``max_passes`` (16). Returns
    the per-width table, the chosen width, whether the pass cap bound at the choice, and the chosen
    :class:`BaselineConfig` (width, heads of ``head_dim``, and ``max_steps`` from
    :func:`steps_for_device_hours`).
    """
    if not tokens_per_s:
        raise ValueError("tokens_per_s must hold at least one ladder width")
    if type(unique_train_tokens) is not int or unique_train_tokens < 1:
        raise ValueError(f"unique_train_tokens must be a positive int, got {unique_train_tokens!r}")
    if tokens_per_parameter <= 0 or max_passes <= 0:
        raise ValueError("tokens_per_parameter and max_passes must be positive")
    tokens_per_step = int(base.batch_size) * int(base.max_seq_len)
    rows: List[Dict[str, Any]] = []
    for width in sorted(int(w) for w in tokens_per_s):
        cfg = base.with_width(width, head_dim=head_dim)
        tau = float(tokens_per_s[width])
        steps = steps_for_device_hours(tau, hours, tokens_per_step)
        affordable = steps * tokens_per_step
        compute_optimal = tokens_per_parameter * cfg.context_processing_parameters()
        cap = max_passes * unique_train_tokens
        required = min(compute_optimal, cap)
        rows.append({
            "width": width, "n_heads": cfg.n_heads, "context_parameters": cfg.context_processing_parameters(),
            "total_parameters": cfg.total_parameters(), "tokens_per_s": tau, "max_steps": steps,
            "affordable_tokens": affordable, "compute_optimal_tokens": compute_optimal,
            "pass_cap_tokens": cap, "required_tokens": required, "cap_binds": cap < compute_optimal,
            "passes": affordable / unique_train_tokens, "fits": affordable >= required,
        })
    fitting = [row for row in rows if row["fits"]]
    chosen = fitting[-1] if fitting else rows[0]
    config = replace(base.with_width(chosen["width"], head_dim=head_dim), max_steps=chosen["max_steps"])
    return {
        "rule": WIDTH_LADDER_RULE, "hours": float(hours), "unique_train_tokens": unique_train_tokens,
        "tokens_per_step": tokens_per_step, "tokens_per_parameter": float(tokens_per_parameter),
        "max_passes": float(max_passes), "candidates": rows, "chosen_width": chosen["width"],
        "chosen_fits": bool(chosen["fits"]), "cap_binds": bool(chosen["cap_binds"]),
        "max_steps": chosen["max_steps"], "config": config,
    }


def total_parameter_width(
    base:         BaselineConfig,
    target_total: int,

    *,
    head_dim:     Optional[int] = None,
) -> Dict[str, Any]:
    r"""The width whose total parameter count is nearest ``target_total`` (Table M2, "Matched total parameters").

    Every field of ``base`` except the width and head count is kept (depth, MLP expansion, protocol,
    optimizer). Over widths :math:`K = m d` in whole heads of :math:`d` = ``head_dim`` (default: the
    base's head width), :math:`m \ge 1`, the total count of :meth:`BaselineConfig.total_parameters`,

    .. math::
        P(K) = L\,(4K^2 + 2eK^2 + eK + K + n_{\rm LN} K) + 2K + 2VK + b V + t\,VKd,

    with :math:`n_{\rm LN} = 4` under pre-LN (0 under ``post_shared``), :math:`b = 1` with a decoder
    bias and :math:`t = 1` with a per-type read map, is strictly increasing in :math:`K`. The chosen
    width minimizes :math:`|P(K) - P^\star|`, the smaller width on a tie, so the rule is deterministic.

    The learning rate, initialization and warmup are the base width's and were not tuned at the chosen
    width. As a declared difference the hidden-matrix learning rate is transferred muP-style (Yang et al.
    2022, arXiv:2203.03466): ``hidden_lr_scale`` becomes :math:`s_0 K_0 / K` with :math:`K_0` and
    :math:`s_0` the base's width and scale, so Adam updates of :math:`W_{Q,K,V,O}` and :math:`W_{1,2}`
    keep their base-width effect on the hidden features; the embedding, decoder, initialization and
    warmup are not transferred.
    Returns the chosen config, ``width``, ``head_dim``, ``total_parameters``, ``target_total``,
    ``relative_error`` :math:`= (P(K) - P^\star) / P^\star` and ``hidden_lr_scale``.
    """
    if type(target_total) is not int or target_total < 1:
        raise ValueError(f"target_total must be a positive int, got {target_total!r}")
    d = base.d_head if head_dim is None else int(head_dim)
    if d < 1:
        raise ValueError(f"head_dim must be >= 1, got {d}")
    m = 1
    while base.with_width(m * d, head_dim=d).total_parameters() < target_total:
        m += 1
    candidates = [base.with_width(k * d, head_dim=d) for k in ((m - 1, m) if m > 1 else (m,))]
    chosen = min(candidates, key=lambda cfg: (abs(cfg.total_parameters() - target_total), cfg.embed_dim))
    chosen = replace(chosen, hidden_lr_scale=base.hidden_lr_scale * base.embed_dim / chosen.embed_dim)
    total = chosen.total_parameters()
    return {"config": chosen, "width": chosen.embed_dim, "head_dim": d, "total_parameters": total,
            "target_total": target_total, "relative_error": (total - target_total) / target_total,
            "hidden_lr_scale": chosen.hidden_lr_scale}


def calibrated_tokens_per_s(
    run_dir: Union[str, Path],

    *,
    column:    str = "train_step_tokens_per_s",
    statistic: str = "median",
) -> Dict[str, Any]:
    r"""Measured training throughput of a (short calibration) run, read from its ``metrics.csv``.

    Uses :func:`vfe3.run_resources.throughput_summary`; the default is the median over the logged
    windows of ``train_step_tokens_per_s`` (optimizer steps only, the quantity whose inverse
    :func:`measured_train_device_hours` integrates), which is robust to the first warm-up window.
    """
    from vfe3.run_resources import throughput_summary

    summary = throughput_summary(run_dir)
    quantiles = (summary.get("throughput_tokens_per_s") or {}).get(column) or {}
    value = quantiles.get(statistic)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f"{run_dir}: no finite {statistic} of {column!r} in metrics.csv")
    device_name = summary.get("device_name")
    if device_name is None:                       # a calibration run writes no provenance.json
        record = Path(run_dir) / CALIBRATION_RECORD
        if record.is_file():
            device_name = json.loads(record.read_text(encoding="utf-8")).get("device_name")
    return {"tokens_per_s": float(value), "column": column, "statistic": statistic,
            "n_windows": quantiles.get("n"), "device_name": device_name,
            "run_dir": str(run_dir)}


# The calibration run's device record (train_baseline.calibrate_widths writes it; a calibration run is
# not finalized, so it has no provenance.json).
CALIBRATION_RECORD: str = "calibration.json"
# Fields a calibration run sets by its own protocol; every other field must equal the matched arm's.
CALIBRATION_PROTOCOL_FIELDS: Tuple[str, ...] = (
    "seed", "max_steps", "warmup_steps", "log_interval", "eval_interval", "checkpoint_interval",
    "resume_from")


def calibration_mismatch(cfg: BaselineConfig, run_dir: Union[str, Path], width: int) -> List[str]:
    r"""Fields in which the calibration run ``run_dir`` differs from ``cfg.with_width(width)``.

    :data:`CALIBRATION_PROTOCOL_FIELDS` are excluded. An empty list means the run measured the
    throughput of the configuration the width rule will train; a config that cannot be read raises
    ``ValueError``.
    """
    path = Path(run_dir) / "config.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    calibrated = BaselineConfig.from_serialized(record.get("config", record), source=str(path))
    expected = cfg.with_width(int(width))
    return [f.name for f in fields(BaselineConfig)
            if f.name not in CALIBRATION_PROTOCOL_FIELDS
            and getattr(calibrated, f.name) != getattr(expected, f.name)]


def measured_train_device_hours(run_dir: Union[str, Path]) -> Dict[str, Any]:
    r"""Training-only device-hours of a finished run: :math:`\sum_r \bar t_r \Delta s_r / 3600`.

    The sum over logging windows of the mean optimizer-step time times the steps in the window
    (:func:`vfe3.run_resources.throughput_summary`), which excludes evaluation and artifact writing;
    the wall time is returned beside it.
    """
    from vfe3.run_resources import throughput_summary

    summary = throughput_summary(run_dir)
    hours = summary.get("train_only_hours_estimate")
    if not isinstance(hours, (int, float)) or not math.isfinite(float(hours)) or hours <= 0:
        raise ValueError(f"{run_dir}: metrics.csv yields no training-time estimate")
    return {"train_hours": float(hours), "wall_time_hours": summary.get("wall_time_hours"),
            "device_name": summary.get("device_name"), "run_dir": str(run_dir)}
