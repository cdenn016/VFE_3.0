r"""Click-to-run: train the main-results arms of GL(K)_attention_v2 (Table tab:placeholder_main_results).

Edit ``CONFIG`` below and run ``python paper_arms.py``. There is no argument parsing. With
``dry_run=True`` (the default) nothing is trained: every arm of every selected base is resolved and
validated for every seed, and the plan is printed (status, parameter counts, token budget, estimated
hours at the base run's measured throughput, estimated disk and training state).

BASES. Each arm is a declared change of one base configuration, read from the base run's
``config.json`` with :func:`vfe3.legacy_config.lenient_config_from_serialized`. Fields a newer code
line retired are dropped and recorded; fields the record lacks take the current defaults and are
listed; the three fields whose defaults the working-tree ``vfe3/config.py`` flips
(:data:`WIP_FLIPPED_DEFAULTS`) are pinned to the run's recorded values (to the pre-flip defaults when
a record lacks one), so the arms never inherit a default flip. Every base is its own experiment with
its own directory and manifest.

BUDGET. ``base_protocol``: the base run's own ``max_steps`` and ``eval_interval`` (the 34.48 run:
150,000 updates, 614.4M token presentations), with ``checkpoint_interval = max_steps``. ``one_epoch``:
``max_steps = ceil(train tokens / (batch_size x max_seq_len))`` on the arm's
dataset, ``eval_interval = max(1, max_steps // 20)`` (at least 20 validation points for Figure M3
once ``max_steps >= 20``; every evaluation also writes the map snapshot of Figure S1) and
``checkpoint_interval = max_steps`` (one training checkpoint at the end, which holds the optimizer
groups S1 reads). The same budget fields are applied to the base and to every arm, so they never
appear in an arm's difference.

ARMS. One entry per row of the table, with the exact labels ``paper_artifacts.py`` and
:mod:`vfe3.paper` use. A VFE arm declares ``VFE3Config`` overrides (a value, or a function of the
resolved base config) and the fields the config itself coerces as a documented consequence
(``gauge_transport='off'`` forces the frame fields). Resolution asserts that the arm's config
differs from the resolved base in the declared fields and the documented coercions only, and that
every declared value survived validation. A baseline arm declares a ``vfe3.baseline`` spec; its
protocol fields (tokenizer, context, stride, budget, cadence, shape, prior, decoder) must equal the
base's except where declared. Rules:

* Stored-covariance decay. "Decayed" keeps the base's ``sigma_weight_decay``; "without decay" sets
  it to 0. For a base whose decay is already 0 (``full_K64``) the row "Learned frames, stored
  covariance without decay" is marked inapplicable, because the evaluated arm already is that
  configuration, and "Identity frames, decayed stored covariance" is inapplicable, because it would
  need a second change (introducing a decay); "Identity frames, stored covariance without decay" is
  then the single change ``gauge_transport='off'``. The labels are written verbatim into LaTeX, so a
  decay-0.04 run is never filed under a "without decay" row.
* Residual removed: inapplicable when the base has no attention residual.
* LayerNorm and MLP in reference coordinates: ``block_coordinates='reference'`` (the decoder stays
  local; one change). On a base without a LayerNorm it conjugates the MLP alone.
* Intertwiner-constrained block: ``block_mlp_mode='gauge_gate'``, the block (and final) LayerNorm
  replaced by ``'mahalanobis'`` where the base has one (``'none'`` stays), and
  ``decode_coordinates='reference'``.
* Word level: dataset ``wikitext-103-word-eos`` (the published stream, one ``<eos>`` per line), the
  vocabulary size from its sidecar, ``decode_ce_checkpoint='auto_aggregate'``, and the one-epoch budget over
  the word training split. The conventional word arm is matched to the Gaussian word arm.
* Matched measured compute: at run time the target is the mean measured train-only device-hours of
  the retrained evaluated arm's complete runs; every width of the pre-declared ladder
  (``width_ladder`` multiples of the matched width, whole 32-dim heads) gets a
  ``calibration_steps`` calibration run whose median train-step throughput feeds
  ``vfe3.baseline.width_ladder_rule``, which picks the width and ``max_steps``; the rule's record is
  written to ``<arm dir>/width_rule.json``. The table reads the best-validation checkpoint.

* Confound controls (October 7, 2026 review of the 34.48 against 55.10 comparison). A baseline arm
  may declare ``overrides`` (``BaselineConfig`` fields applied after ``baseline_overrides``); a
  protocol field among them is a declared difference. "Per-type read table, ALiBi (as run)" repeats
  the 55.10 configuration (``block_read``, ALiBi); the existing "Per-type table of the frame table's
  size" row, matched to a causal-prior base, is the same table under the realization's uniform causal
  prior. From the as-run arm, one change each: the realization's optimizer (Adam ``beta_2 = 0.999``,
  no decay on matrices, decay 0.1 on the table as on the frames; the realization's 0.01 MLP decay and
  per-role clipping remain different), a read-and-write table (``block_rw``), and the shared post-norm
  layout (``norm_placement='post_shared'``). VFE arms: the realization with the ALiBi prior, and the
  covariance-blind routing cost ``relational_routing='mean_distance'``. :data:`TIER_A` lists the
  baseline controls and :data:`TIER_B` the VFE retraining controls.

An arm whose code is not in this build (a missing config field or ``vfe3.baseline`` /
``train_baseline``) is reported as unavailable with the reason; an arm that does not apply to the
base as inapplicable; an arm whose config the validator rejects as invalid (a training launch
refuses to start while a selected arm is invalid).

SEEDS: ``seeds`` lists ints (``data_seed = seed``) or explicit ``(seed, data_seed)`` pairs, shared
by every arm so replicates are paired across arms. Under the ``base_protocol`` budget the evaluated
arm's replicate whose pair equals the base run's own ``(seed, data_seed)`` is the base run itself: it is
reused (skipped in training, listed in the manifest, and read by the compute-matched rule) rather than
retrained.

LAYOUT: ``<out_root>/<base>/<budget>/<arm slug>/<test PPL>_<slug>_s<seed>_d<data seed>/``. An (arm,
seed) pair whose run directory already holds a complete ``summary.json`` (finite ``test_ce``) is
skipped. After every run the manifest ``<out_root>/<base>/<budget>/m2_manifest.json`` is rewritten
with every row of the table (rows without complete runs stay pending); pass it to
``python paper_artifacts.py <manifest>``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import re
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

ROOT = Path(__file__).resolve().parent

CONFIG: Dict[str, Any] = {
    "bases":             ["k128_exp"],     # keys of BASES
    "arms":              "TIER_A",         # None: every row of ARMS in table order; a list of labels;
                                           # or "TIER_A" / "TIER_B" (the confound controls)
    "budget":            "base_protocol",  # key of BUDGETS
    "seeds":             [(6, 3), (23, 23), (54, 54)],   # (seed, data_seed); (6, 3) is the 34.48 run's
    "out_root":          "vfe3_runs/paper_arms",
    "dry_run":           True,             # True: resolve, validate and print the plan; train nothing
    "device":            None,             # None -> cuda when available
    "calibration_steps": 200,              # matched-compute baseline: steps of each width's calibration run
    "width_ladder":      (1, 1.5, 2, 3, 4, 6, 8),   # matched-compute ladder, multiples of the matched width
    "baseline_overrides": {"lr": 3e-3},    # BaselineConfig optimizer fields for every baseline arm:
                                           # 3e-3 is the validation choice of the October 6 sweep
}

BASES: Dict[str, str] = {
    # The evaluated K=128 exponential-chart run of the paper (34.48 test PPL).
    "k128_exp":  "vfe3_runs/34.48_wikitext-103_K128_block_glk_linear_mlp_s6",
    # Bases of the earlier study (not paper results).
    "diag_K128": "D:/DATA_KEEP/39.77_wikitext-103_K128_n=128",
    "diag_K64":  "D:/DATA_KEEP/52.91_wikitext-103_K64_n=128",
    "full_K64":  "vfe3_runs/Full_Gaussian_mom-match_Template_routing/"
                 "55.07_wikitext-103_K64_moment_matched_noAtnnRes",
}

# Section headings of the table (identical to paper_artifacts.py).
_RETRAINING = "Retraining arms at the evaluated configuration and token budget"
_BASELINES = "Conventional transformers: same tokenizer, context, stride, and budget"
_WORD_LEVEL = "Word-level protocol, every target scored"
_CONFOUNDS = "Confound controls for the per-type read-table comparison"

WORD_DATASET = "wikitext-103-word-eos"
EVALUATED = "Evaluated: learned frames, decayed stored covariance"
M3_POINTS = 20

# Fields whose dataclass defaults the working-tree vfe3/config.py flips, with the defaults they had
# before the flip (used only when a base record lacks the field).
WIP_FLIPPED_DEFAULTS: Dict[str, Any] = {
    "norm_type_block": "layernorm",
    "layernorm_affine": True,
    "skip_belief_sigma_update": True,
}

# Fields VFE3Config.__post_init__ sets when gauge_transport='off' (identity frame, frozen).
GAUGE_OFF_COERCED: Tuple[str, ...] = ("phi_scale", "pos_phi", "m_phi_lr")


# ----------------------------------------------------------------------------------------------
# Arm rules (functions of the resolved base config)
# ----------------------------------------------------------------------------------------------

def _covariant_norm(name: str) -> Callable[[Any], str]:
    r"""``'mahalanobis'`` where the base's ``name`` norm is a LayerNorm, the base value otherwise."""
    def value(base: Any) -> str:
        current = getattr(base, name)
        return "mahalanobis" if current == "layernorm" else current
    return value


def _word_vocab_size(base: Any) -> int:
    r"""Vocabulary of the word dataset, read from its sidecar (the published 267,735 when absent)."""
    from vfe3.data.datasets import tokenizer_vocab_size
    return int(tokenizer_vocab_size(WORD_DATASET))


def _no_decay_already(base: Any) -> Optional[str]:
    if base.sigma_weight_decay == 0:
        return ("the base has sigma_weight_decay=0, so the evaluated arm already is this row's "
                "configuration")
    return None


def _decay_absent(base: Any) -> Optional[str]:
    if base.sigma_weight_decay == 0:
        return ("the base has no stored-covariance decay (sigma_weight_decay=0); this row would need "
                "one as a second change")
    return None


def _no_residual(base: Any) -> Optional[str]:
    return None if base.attention_residual else "the base has no attention residual"


def _no_mlp(base: Any) -> Optional[str]:
    return None if base.use_block_mlp else "the base has no block MLP"


def _no_mlp_or_reference_decode(base: Any) -> Optional[str]:
    r"""The intertwiner arm decodes in reference coordinates, exact only for the Gaussian-KL decoders."""
    from vfe3.config import _REFERENCE_DECODE_MODES

    if not base.use_block_mlp:
        return "the base has no block MLP"
    if base.decode_mode not in _REFERENCE_DECODE_MODES:
        return (f"decode_mode={base.decode_mode!r} has no reference-coordinate decode (the reduction to "
                f"(xi, C) is exact only for {_REFERENCE_DECODE_MODES})")
    return None


def _no_coordinate_mlp(base: Any) -> Optional[str]:
    if not base.use_block_mlp or base.block_mlp_mode != "coordinate":
        return "the per-type hidden bias needs the base's coordinate block MLP"
    return None


def _template_already(base: Any) -> Optional[str]:
    if not base.use_block_mlp:
        return "the base has no block MLP"
    if base.block_mlp_mode == "gaussian_template":
        return "the base MLP already is Gaussian template routing"
    return None


def _no_ln_or_mlp(base: Any) -> Optional[str]:
    if base.norm_type_block != "layernorm" and base.norm_type_final != "layernorm" and not base.use_block_mlp:
        return "the base has neither a LayerNorm nor a block MLP"
    return None


def _word_level_base(base_dataset: str) -> Optional[str]:
    from vfe3.data.datasets import is_word_level
    return "the base is already word-level" if is_word_level(base_dataset) else None


# Every row of tab:placeholder_main_results, in table order. Keys per arm:
#   group, slug           section heading; run-directory name
#   vfe                   VFE3Config overrides (value, or function of the resolved base config)
#   coerced               fields the config sets as a documented consequence of the overrides
#   requires              VFE3Config fields the arm needs (unavailable when this build lacks one)
#   inapplicable          function(base cfg) -> reason, or None when the arm applies
#   baseline              vfe3.baseline spec: {"per_type_table": ..., "compute_matched": bool}
#   dataset               dataset override (word level)
#   matched_to_arm        baseline matched to another arm's resolved config instead of the base
def _alibi_already(cfg: Any) -> Optional[str]:
    if getattr(cfg, "beta_attention_prior", None) == "causal_alibi":
        return "the base already uses the ALiBi prior"
    return None


def _no_kernel_routing(cfg: Any) -> Optional[str]:
    if getattr(cfg, "attention_mode", None) != "projected_predictive":
        return "mean_distance routing requires attention_mode='projected_predictive'"
    if getattr(cfg, "predictive_readout", "kernel") != "kernel":
        return "mean_distance routing requires predictive_readout='kernel'"
    if getattr(cfg, "relational_routing", "gaussian_kl") != "gaussian_kl":
        return "the base does not route by the Gaussian KL"
    return None


# The ALiBi confound swaps the positional scheme: on a RoPE base the rotation is removed with it, so the
# arm is ALiBi instead of RoPE, not ALiBi on top of RoPE (a no-op change on a base without RoPE).
_ALIBI = {"beta_attention_prior": "causal_alibi", "pos_rotation": "none"}
# The realization's optimizer on the baseline: Adam beta_2 0.999 and no decay on matrices (the prior
# means, comparison/content maps and decoder are undecayed), decay 0.1 on the table as on the frames.
_REALIZATION_OPTIMIZER = {"adam_beta2": 0.999, "weight_decay": 0.0, "per_type_table_weight_decay": 0.1}

ARMS: Dict[str, Dict[str, Any]] = {
    EVALUATED: {
        "group": _RETRAINING, "slug": "evaluated", "vfe": {}},
    "Learned frames, stored covariance without decay": {
        "group": _RETRAINING, "slug": "no_sigma_decay", "vfe": {"sigma_weight_decay": 0.0},
        "inapplicable": _no_decay_already},
    "Identity frames, decayed stored covariance": {
        "group": _RETRAINING, "slug": "identity_frames", "vfe": {"gauge_transport": "off"},
        "coerced": GAUGE_OFF_COERCED, "inapplicable": _decay_absent},
    "Identity frames, stored covariance without decay": {
        "group": _RETRAINING, "slug": "identity_frames_no_decay",
        "vfe": {"gauge_transport": "off", "sigma_weight_decay": 0.0}, "coerced": GAUGE_OFF_COERCED},
    "Identity frames, per-type additive table of equal size": {
        "group": _RETRAINING, "slug": "identity_frames_type_bias",
        "vfe": {"gauge_transport": "off", "per_type_hidden_bias": True},
        # m_per_type_hidden_bias_lr None resolves to the declared m_phi_lr (read before the coercion).
        "coerced": (*GAUGE_OFF_COERCED, "m_per_type_hidden_bias_lr"),
        "requires": ("per_type_hidden_bias", "m_per_type_hidden_bias_lr"),
        "inapplicable": _no_coordinate_mlp},
    "Attention residual removed": {
        "group": _RETRAINING, "slug": "no_residual", "vfe": {"attention_residual": False},
        "inapplicable": _no_residual},
    "LayerNorm and MLP in reference coordinates": {
        "group": _RETRAINING, "slug": "reference_block", "vfe": {"block_coordinates": "reference"},
        "requires": ("block_coordinates",), "inapplicable": _no_ln_or_mlp},
    "Intertwiner-constrained nonlinear block": {
        "group": _RETRAINING, "slug": "intertwiner_block",
        "vfe": {"block_mlp_mode": "gauge_gate",
                "norm_type_block": _covariant_norm("norm_type_block"),
                "norm_type_final": _covariant_norm("norm_type_final"),
                "decode_coordinates": "reference"},
        "requires": ("decode_coordinates",), "inapplicable": _no_mlp_or_reference_decode},
    "MLP removed": {
        "group": _RETRAINING, "slug": "no_mlp", "vfe": {"use_block_mlp": False},
        "inapplicable": _no_mlp},
    "MLP replaced by Gaussian template routing": {
        "group": _RETRAINING, "slug": "template_mlp", "vfe": {"block_mlp_mode": "gaussian_template"},
        "inapplicable": _template_already},
    "Matched context-processing parameters": {
        "group": _BASELINES, "slug": "tf_matched_params", "baseline": {"per_type_table": "none"}},
    "Matched measured compute": {
        "group": _BASELINES, "slug": "tf_matched_compute",
        "baseline": {"per_type_table": "none", "compute_matched": True}},
    "Per-type table of the frame table's size": {
        "group": _BASELINES, "slug": "tf_type_table", "baseline": {"per_type_table": "block_read"}},
    "Learned frames, ALiBi prior": {
        "group": _CONFOUNDS, "slug": "alibi_prior", "vfe": dict(_ALIBI), "inapplicable": _alibi_already},
    "Covariance-blind routing (mean distance)": {
        "group": _CONFOUNDS, "slug": "mean_distance", "vfe": {"relational_routing": "mean_distance"},
        "requires": ("relational_routing",), "inapplicable": _no_kernel_routing},
    "Per-type read table, ALiBi (as run)": {
        "group": _CONFOUNDS, "slug": "tf_type_table_alibi",
        "baseline": {"per_type_table": "block_read", "overrides": dict(_ALIBI)}},
    "Per-type read table, ALiBi, realization optimizer": {
        "group": _CONFOUNDS, "slug": "tf_type_table_alibi_opt",
        "baseline": {"per_type_table": "block_read", "overrides": {**_ALIBI, **_REALIZATION_OPTIMIZER}}},
    "Per-type read-and-write table, ALiBi": {
        "group": _CONFOUNDS, "slug": "tf_type_table_rw_alibi",
        "baseline": {"per_type_table": "block_rw", "overrides": dict(_ALIBI)}},
    "Per-type read table, ALiBi, shared post-norm": {
        "group": _CONFOUNDS, "slug": "tf_type_table_alibi_postnorm",
        "baseline": {"per_type_table": "block_read", "overrides": {**_ALIBI, "norm_placement": "post_shared"}}},
    "Gaussian attention": {
        "group": _WORD_LEVEL, "slug": "word_gaussian", "dataset": WORD_DATASET,
        "vfe": {"vocab_size": _word_vocab_size, "decode_ce_checkpoint": "auto_aggregate"}},
    "Conventional transformer, matched": {
        "group": _WORD_LEVEL, "slug": "word_tf_matched", "dataset": WORD_DATASET,
        "baseline": {"per_type_table": "none"}, "matched_to_arm": "Gaussian attention"},
}

# Baseline protocol fields that must equal the matched VFE config (BaselineConfig name -> VFE3Config
# name); only fields present on both are compared.
BASELINE_PROTOCOL_FIELDS: Dict[str, str] = {
    "vocab_size": "vocab_size", "max_seq_len": "max_seq_len", "eval_stride": "eval_stride",
    "batch_size": "batch_size", "max_steps": "max_steps", "eval_interval": "eval_interval",
    "checkpoint_interval": "checkpoint_interval", "grad_accum_steps": "grad_accum_steps",
    "embed_dim": "embed_dim", "n_heads": "n_heads", "n_layers": "n_layers",
    "mlp_expansion": "block_mlp_expansion", "seed": "seed", "deterministic": "deterministic",
    "amp_dtype": "amp_dtype", "fp32_matmul_precision": "fp32_matmul_precision",
    "warmup_steps": "warmup_steps", "min_lr_frac": "min_lr_frac",
    "grad_clip": "grad_clip", "beta_attention_prior": "beta_attention_prior",
    "alibi_slope": "alibi_slope", "pos_rotation": "pos_rotation", "rope_base": "rope_base",
    "decode_bias": "decode_bias",
    "decode_unigram_prior": "decode_unigram_prior", "unigram_kappa": "unigram_kappa",
}
# Confound controls for the 34.48 against 55.10 comparison (October 7, 2026). TIER_A: conventional
# baselines (about 1.5 GPU-hours each at the base protocol, the compute-matched arm about as long as
# the evaluated run). TIER_B: VFE retraining arms (about 16 GPU-hours each).
TIER_A: Tuple[str, ...] = (
    "Per-type read table, ALiBi (as run)",
    "Per-type table of the frame table's size",
    "Per-type read table, ALiBi, realization optimizer",
    "Per-type read-and-write table, ALiBi",
    "Per-type read table, ALiBi, shared post-norm",
    "Matched measured compute",
)
TIER_B: Tuple[str, ...] = (
    EVALUATED,
    "Identity frames, stored covariance without decay",
    "Identity frames, per-type additive table of equal size",
    "Learned frames, ALiBi prior",
    "Covariance-blind routing (mean distance)",
)

# Fields the compute-matched baseline changes by construction (width rule, calibrated steps, cadence).
COMPUTE_MATCHED_FIELDS: Tuple[str, ...] = (
    "embed_dim", "n_heads", "max_steps", "eval_interval", "checkpoint_interval")


# ----------------------------------------------------------------------------------------------
# Bases and budgets
# ----------------------------------------------------------------------------------------------

class ArmDefinitionError(AssertionError):
    r"""An arm's resolved config differs from the base in an undeclared way (a defect of ARMS)."""


@dataclass
class Base:
    r"""One base run: its loaded config and the facts the plan reports about it."""

    name:             str
    run_dir:          Path
    cfg:              Any                               # VFE3Config
    dataset:          str
    dropped:          Dict[str, Any] = field(default_factory=dict)   # retired fields dropped on load
    pinned:           Dict[str, Any] = field(default_factory=dict)   # WIP-flipped fields -> value used
    defaulted:        List[str] = field(default_factory=list)        # fields the record lacks
    recorded_params:  Optional[int] = None                           # config.json n_params
    tokens_per_s:     Optional[float] = None                         # measured wall-clock throughput
    throughput_source: Optional[str] = None
    data_seed:        Optional[int] = None                           # provenance.json data-order seed


def _quiet_config(build: Callable[[], Any]) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build()


def load_base(name: str, path: Union[str, Path]) -> Base:
    r"""Load a base run's config (lenient, WIP-flipped fields pinned) and its measured throughput."""
    from vfe3.config import VFE3Config
    from vfe3.legacy_config import lenient_config_from_serialized

    run_dir = Path(path)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    record = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    raw = dict(record.get("config", record))
    pinned = {key: raw.get(key, default) for key, default in WIP_FLIPPED_DEFAULTS.items()}
    payload = {**raw, **pinned}
    cfg, dropped = _quiet_config(
        lambda: lenient_config_from_serialized(payload, source=str(run_dir / "config.json")))
    present = set(payload)
    if present & {"relational_mode", "relational_readout_mode"}:
        present.add("attention_mode")                     # migrated from the historical pair
    defaulted = sorted(f.name for f in dataclasses.fields(VFE3Config) if f.name not in present)
    dataset = record.get("dataset")
    if not isinstance(dataset, str) or not dataset:
        raise ValueError(f"{run_dir / 'config.json'} records no dataset")
    base = Base(name=name, run_dir=run_dir, cfg=cfg, dataset=dataset, dropped=dict(dropped),
                pinned=pinned, defaulted=defaulted,
                recorded_params=record.get("n_params") if isinstance(record.get("n_params"), int) else None)
    provenance_path = run_dir / "provenance.json"
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        seed_value = provenance.get("data_seed") if isinstance(provenance, dict) else None
        if isinstance(seed_value, int) and not isinstance(seed_value, bool):
            base.data_seed = seed_value
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        steps, wall = summary.get("n_steps"), summary.get("wall_time_s")
        if isinstance(steps, int) and isinstance(wall, (int, float)) and wall > 0:
            base.tokens_per_s = steps * cfg.batch_size * cfg.max_seq_len / float(wall)
            base.throughput_source = "summary.json n_steps x batch x context / wall_time_s (incl. evaluation)"
    return base


def one_epoch(cfg: Any, dataset: str) -> Dict[str, int]:
    r"""One pass over the training split at the config's batch and context.

    :math:`S = \lceil T_{\mathrm{train}} / (B N) \rceil`, ``eval_interval = max(1, S // 20)`` and
    ``checkpoint_interval = S``.
    """
    from vfe3.data.datasets import cached_token_count

    tokens = int(cached_token_count(dataset, "train"))
    per_step = int(cfg.batch_size) * int(cfg.max_seq_len)
    max_steps = -(-tokens // per_step)
    return {"max_steps": max_steps, "eval_interval": max(1, max_steps // M3_POINTS),
            "checkpoint_interval": max_steps}


def base_protocol(cfg: Any, dataset: str) -> Dict[str, int]:
    r"""The base run's own step budget and validation cadence, one checkpoint at the end."""
    max_steps = int(cfg.max_steps)
    return {"max_steps": max_steps, "eval_interval": int(cfg.eval_interval), "checkpoint_interval": max_steps}


BUDGETS: Dict[str, Callable[[Any, str], Dict[str, int]]] = {"one_epoch": one_epoch, "base_protocol": base_protocol}


def rebuild(cfg: Any, overrides: Mapping[str, Any], source: str) -> Any:
    r"""``cfg`` with ``overrides`` applied, rebuilt (and validated) through the serialized form."""
    from vfe3.config import config_from_serialized, serialize_config

    return _quiet_config(lambda: config_from_serialized({**serialize_config(cfg), **overrides}, source=source))


def resolved_base(base: Base, budget: str, dataset: str) -> Any:
    r"""The base config with the budget of ``budget`` on ``dataset`` applied."""
    return rebuild(base.cfg, BUDGETS[budget](base.cfg, dataset), f"{base.name} + {budget} on {dataset}")


def config_difference(before: Any, after: Any) -> Dict[str, Tuple[Any, Any]]:
    r"""``{field: (before, after)}`` for every serialized field that differs."""
    from vfe3.config import serialize_config

    a, b = serialize_config(before), serialize_config(after)
    return {key: (a.get(key), b.get(key)) for key in sorted(set(a) | set(b)) if a.get(key) != b.get(key)}


def check_vfe_arm(
    base_cfg: Any,
    arm_cfg:  Any,
    declared: Mapping[str, Any],
    coerced:  Sequence[str] = (),
) -> Dict[str, Tuple[Any, Any]]:
    r"""Assert ``arm_cfg`` differs from ``base_cfg`` only in ``declared`` and ``coerced``; return the diff.

    Every declared value must also be the resolved value (a validator that silently rewrote an
    override would change the arm), and an arm with declared overrides must differ from the base.
    """
    diff = config_difference(base_cfg, arm_cfg)
    undeclared = sorted(set(diff) - set(declared) - set(coerced))
    if undeclared:
        raise ArmDefinitionError(
            "the resolved arm differs from the base in undeclared field(s) "
            + ", ".join(f"{key}: {diff[key][0]!r} -> {diff[key][1]!r}" for key in undeclared))
    rewritten = sorted(key for key, value in declared.items() if getattr(arm_cfg, key) != value)
    if rewritten:
        raise ArmDefinitionError(
            "declared value(s) did not survive validation: "
            + ", ".join(f"{key}={declared[key]!r} resolved to {getattr(arm_cfg, key)!r}" for key in rewritten))
    if declared and not diff:
        raise ArmDefinitionError("the arm resolves to the base configuration itself")
    return diff


def check_baseline_arm(
    matched_cfg:  Any,
    baseline_cfg: Any,
    declared:     Sequence[str] = (),
) -> Dict[str, Tuple[Any, Any]]:
    r"""Assert the baseline's protocol fields equal the matched VFE config's except ``declared``.

    Returns ``{BaselineConfig field: (VFE value, baseline value)}`` of the declared differences.
    """
    declared = set(declared)
    if getattr(matched_cfg, "decode_mode", "linear") != "linear":
        declared.add("decode_bias")         # documented: the baseline decodes linearly, with a bias
    diff: Dict[str, Tuple[Any, Any]] = {}
    for name, vfe_name in BASELINE_PROTOCOL_FIELDS.items():
        if not hasattr(baseline_cfg, name) or not hasattr(matched_cfg, vfe_name):
            continue
        a, b = getattr(matched_cfg, vfe_name), getattr(baseline_cfg, name)
        if a != b:
            diff[name] = (a, b)
    undeclared = sorted(set(diff) - declared)
    if undeclared:
        raise ArmDefinitionError(
            "the baseline's protocol differs from the matched configuration in "
            + ", ".join(f"{key}: {diff[key][0]!r} -> {diff[key][1]!r}" for key in undeclared))
    if getattr(baseline_cfg, "model_family", None) != "conventional_transformer":
        raise ArmDefinitionError(f"baseline model_family is {getattr(baseline_cfg, 'model_family', None)!r}")
    return diff


# ----------------------------------------------------------------------------------------------
# Parameter counts (exact, from two small-vocabulary builds)
# ----------------------------------------------------------------------------------------------

_PROBE_VOCAB: Tuple[int, int] = (64, 128)


def _counts(model: Any, cfg: Any) -> Dict[str, int]:
    r"""Reported counts (:func:`vfe3.run_resources.parameter_breakdown`, frozen tables excluded) and
    the stored and trainable totals."""
    from vfe3.run_resources import parameter_breakdown

    params = dict(model.named_parameters())
    breakdown = parameter_breakdown(params, cfg)
    return {"total": int(breakdown["total"]),
            "stored": sum(int(p.numel()) for p in params.values()),
            "trainable": sum(int(p.numel()) for p in params.values() if p.requires_grad),
            "context_processing": int(breakdown["context_processing"]["total"]),
            "per_type_tables": int(breakdown["per_type_tables"]["total"])}


def parameter_counts(cfg: Any, build: Callable[[Any], Any], with_vocab: Callable[[Any, int], Any]) -> Dict[str, int]:
    r"""Exact parameter counts at ``cfg.vocab_size`` from builds at two small vocabularies.

    Every vocabulary-sized table is linear in :math:`V` and nothing else depends on it, so each count
    is :math:`P(V) = P(V_1) + (V - V_1)(P(V_2) - P(V_1)) / (V_2 - V_1)`. This avoids materializing
    the full per-type tables (1.2B parameters for the word arms) on the CPU.
    """
    import torch

    v1, v2 = _PROBE_VOCAB
    with torch.no_grad():
        small = [_counts(_quiet_config(lambda v=v: build(with_vocab(cfg, v))), with_vocab(cfg, v)) for v in (v1, v2)]
    V = int(cfg.vocab_size)
    out = {}
    for key in small[0]:
        slope, rem = divmod(small[1][key] - small[0][key], v2 - v1)
        if rem:
            raise ArithmeticError(f"{key} count is not linear in the vocabulary size")
        out[key] = small[0][key] + (V - v1) * slope
    return out


def _vfe_counts(cfg: Any) -> Dict[str, int]:
    from vfe3.model.model import VFEModel
    return parameter_counts(cfg, VFEModel, lambda c, v: rebuild(c, {"vocab_size": v}, "parameter probe"))


def _baseline_counts(cfg: Any) -> Dict[str, int]:
    from vfe3.baseline import BaselineTransformer
    return parameter_counts(cfg, BaselineTransformer, lambda c, v: dataclasses.replace(c, vocab_size=v))


# ----------------------------------------------------------------------------------------------
# Arm resolution and the plan
# ----------------------------------------------------------------------------------------------

@dataclass
class ArmPlan:
    r"""One arm resolved against one base."""

    label:        str
    group:        str
    slug:         str
    kind:         str                                   # "vfe" | "baseline"
    status:       str = "ready"                         # ready | deferred | inapplicable | unavailable | invalid
    reason:       Optional[str] = None
    dataset:      Optional[str] = None
    cfg:          Any = None                            # VFE3Config or BaselineConfig (base seed)
    matched_cfg:  Any = None                            # baseline arms: the VFE config it is matched to
    diff:         Dict[str, Tuple[Any, Any]] = field(default_factory=dict)
    seeds:        List[Tuple[int, int]] = field(default_factory=list)
    params:       Optional[Dict[str, int]] = None
    tokens:       Optional[int] = None                  # training-token presentations per run
    est_hours:    Optional[float] = None
    notes:        List[str] = field(default_factory=list)
    reused:       List[Tuple[int, int]] = field(default_factory=list)   # seed pairs served by the base run


def seed_pairs(seeds: Sequence[Union[int, Sequence[int]]]) -> List[Tuple[int, int]]:
    r"""``[(seed, data_seed)]``: an int means ``data_seed = seed``; pairs are taken as given."""
    out: List[Tuple[int, int]] = []
    for entry in seeds:
        if isinstance(entry, int):
            pair: Tuple[Any, ...] = (entry, entry)
        elif isinstance(entry, (tuple, list)):
            pair = tuple(entry)
        else:
            raise ValueError(f"seed entry {entry!r} must be a non-negative int or a (seed, data_seed) pair")
        if len(pair) != 2 or any(type(v) is not int or v < 0 for v in pair):
            raise ValueError(f"seed entry {entry!r} must be a non-negative int or a (seed, data_seed) pair")
        out.append((int(pair[0]), int(pair[1])))
    if len(set(out)) != len(out):
        raise ValueError(f"duplicate (seed, data_seed) pairs in {list(seeds)!r}")
    return out


def _evaluate(value: Any, base_cfg: Any) -> Any:
    return value(base_cfg) if callable(value) else value


def _baseline_api() -> Any:
    import vfe3.baseline as api
    for name in ("BaselineConfig", "BaselineTransformer", "steps_for_device_hours", "width_ladder_rule",
                 "calibrated_tokens_per_s", "measured_train_device_hours"):
        if not hasattr(api, name):
            raise ImportError(f"vfe3.baseline has no {name}")
    return api


def _run_baseline_api() -> Callable[..., Path]:
    from train_baseline import run_baseline
    return run_baseline


def _with_seed(cfg: Any, seed: int) -> Any:
    r"""``cfg`` at ``seed`` (a VFE config is rebuilt and revalidated; a baseline config replaced)."""
    from vfe3.config import VFE3Config

    if isinstance(cfg, VFE3Config):
        return rebuild(cfg, {"seed": seed}, "seed override")
    return dataclasses.replace(cfg, seed=seed)


def resolve_arm(
    base:     Base,
    label:    str,
    spec:     Mapping[str, Any],

    *,
    budget:   str,
    seeds:    Sequence[Tuple[int, int]],
    resolved: Optional[Mapping[str, ArmPlan]] = None,   # already resolved arms (matched_to_arm)
    counts:   bool = True,
    baseline_overrides: Optional[Mapping[str, Any]] = None,
) -> ArmPlan:
    r"""Resolve one arm against ``base``: status, config, validated difference, counts and cost."""
    from vfe3.config import VFE3Config

    kind = "baseline" if "baseline" in spec else "vfe"
    plan = ArmPlan(label=label, group=spec["group"], slug=spec["slug"], kind=kind,
                   dataset=spec.get("dataset", base.dataset), seeds=list(seeds))
    known = {f.name for f in dataclasses.fields(VFE3Config)}
    missing = [name for name in spec.get("requires", ()) if name not in known]
    if missing:
        plan.status, plan.reason = "unavailable", f"VFE3Config has no field(s) {missing} in this build"
        return plan
    reason = spec["inapplicable"](base.cfg) if spec.get("inapplicable") else None
    if reason is None and "dataset" in spec:
        reason = _word_level_base(base.dataset)
    if reason is not None:
        plan.status, plan.reason = "inapplicable", reason
        return plan
    try:
        base_cfg = resolved_base(base, budget, plan.dataset)
    except FileNotFoundError as exc:
        plan.status, plan.reason = "unavailable", f"dataset cache missing: {exc}"
        return plan
    if kind == "vfe":
        declared = {key: _evaluate(value, base_cfg) for key, value in spec["vfe"].items()}
        try:
            plan.cfg = rebuild(base_cfg, declared, f"{base.name}: {label}")
        except (ValueError, TypeError) as exc:
            plan.status, plan.reason = "invalid", f"config rejected: {exc}"
            return plan
        plan.diff = check_vfe_arm(base_cfg, plan.cfg, declared, spec.get("coerced", ()))
        inert = plan.cfg.inactive_fields()
        plan.notes.extend(f"declared {key}={declared[key]!r} is inert here: {inert[key]}"
                          for key in declared if key in inert and key in plan.diff)
        for seed, _ in seeds:
            _with_seed(plan.cfg, seed)                      # validates every seed's config
        if counts:
            plan.params = _vfe_counts(plan.cfg)
    else:
        _resolve_baseline(spec, plan, base_cfg, resolved or {}, counts=counts,
                          baseline_overrides=baseline_overrides)
        if plan.status not in ("ready", "deferred"):
            return plan
    cfg = plan.cfg
    plan.tokens = int(cfg.max_steps) * int(cfg.batch_size) * int(cfg.max_seq_len)
    if plan.kind == "vfe" and base.tokens_per_s:
        plan.est_hours = plan.tokens / base.tokens_per_s / 3600.0
    return plan


def _resolve_baseline(
    spec:     Mapping[str, Any],
    plan:     ArmPlan,
    base_cfg: Any,
    resolved: Mapping[str, ArmPlan],

    *,
    counts:   bool,
    baseline_overrides: Optional[Mapping[str, Any]] = None,
) -> None:
    r"""Resolve a baseline arm: ``BaselineConfig.matched_to`` the base (or another arm) and check its protocol."""
    try:
        api = _baseline_api()
        _run_baseline_api()
    except ImportError as exc:
        plan.status, plan.reason = "unavailable", f"baseline code missing: {exc}"
        return
    matched = base_cfg
    target = spec.get("matched_to_arm")
    if target is not None:
        other = resolved.get(target)
        if other is None or other.status != "ready":
            plan.status = "inapplicable" if other is not None and other.status == "inapplicable" else "unavailable"
            plan.reason = f"matched to {target!r}, which is {other.status if other else 'unresolved'}"
            return
        matched = other.cfg
    options = spec["baseline"]
    arm_overrides = dict(options.get("overrides", {}))
    try:
        plan.cfg = api.BaselineConfig.matched_to(matched, per_type_table=options.get("per_type_table", "none"),
                                                 **{**dict(baseline_overrides or {}), **arm_overrides})
    except (ValueError, TypeError) as exc:
        plan.status, plan.reason = "invalid", f"BaselineConfig.matched_to rejected the base: {exc}"
        return
    plan.matched_cfg = matched
    # A protocol field set by the arm's own overrides is a declared difference.
    plan.diff = check_baseline_arm(matched, plan.cfg, declared=tuple(arm_overrides))
    for seed, _ in plan.seeds:
        _with_seed(plan.cfg, seed)
    if options.get("compute_matched"):
        plan.status = "deferred"
        plan.reason = ("width and max_steps are set at run time by vfe3.baseline.width_ladder_rule from the "
                       "retrained evaluated arm's measured train device-hours and one calibration run per "
                       "ladder width; shown at the matched width and budget")
    if counts:
        plan.params = _baseline_counts(plan.cfg)


def build_plan(
    base:   Base,

    *,
    budget: str,
    seeds:  Sequence[Tuple[int, int]],
    labels: Optional[Sequence[str]] = None,
    arms:   Optional[Mapping[str, Mapping[str, Any]]] = None,
    counts: bool = True,
    baseline_overrides: Optional[Mapping[str, Any]] = None,
) -> List[ArmPlan]:
    r"""Resolve every selected arm against ``base`` (in table order)."""
    arms = ARMS if arms is None else arms
    selected = list(arms) if labels is None else list(labels)
    unknown = [label for label in selected if label not in arms]
    if unknown:
        raise ValueError(f"unknown arm label(s) {unknown}")
    resolved: Dict[str, ArmPlan] = {}
    for label in arms:
        needed = label in selected or any(arms[other].get("matched_to_arm") == label for other in selected)
        if needed:
            resolved[label] = resolve_arm(base, label, arms[label], budget=budget, seeds=seeds,
                                          resolved=resolved, counts=counts,
                                          baseline_overrides=baseline_overrides)
    plans = [resolved[label] for label in arms if label in selected]
    evaluated = resolved.get(EVALUATED)
    if evaluated is not None and base.cfg.sigma_weight_decay == 0:
        evaluated.notes.append("the row label says 'decayed stored covariance' but this base has "
                               "sigma_weight_decay=0")
    if evaluated is not None and evaluated.params and base.recorded_params is not None:
        total = evaluated.params["total"]
        evaluated.notes.append(
            f"total parameters {total:,} equal the base record" if total == base.recorded_params
            else f"total parameters {total:,} differ from the base record {base.recorded_params:,}")
    if (evaluated is not None and budget == "base_protocol" and base.data_seed is not None
            and (int(base.cfg.seed), base.data_seed) in evaluated.seeds and is_complete(base.run_dir)):
        evaluated.reused = [(int(base.cfg.seed), base.data_seed)]
        evaluated.notes.append(f"seed pair {evaluated.reused[0]} is the base run itself and is reused, "
                               "not retrained")
    for plan in plans:
        if plan.status == "deferred" and evaluated is not None:
            plan.est_hours = evaluated.est_hours
        if plan.kind == "vfe" and plan.dataset != base.dataset and plan.est_hours is not None:
            plan.notes.append(f"est. hours use the base's {base.dataset} throughput; the vocabulary-sized "
                              "decode and table cost of this dataset is not included")
    return plans


def _fmt_count(value: Optional[int]) -> str:
    if value is None:
        return "-"
    for unit, scale in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(value) >= scale:
            return f"{value / scale:.2f}{unit}"
    return str(value)


def format_plan(base: Base, plans: Sequence[ArmPlan], *, budget: str) -> str:
    r"""The plan table for one base (plain text)."""
    cfg = base.cfg
    lines = [
        "=" * 118,
        f"base {base.name}: {base.run_dir}",
        f"  dataset={base.dataset} K={cfg.embed_dim} heads={cfg.n_heads} L={cfg.n_layers} family={cfg.family} "
        f"attention={cfg.attention_mode} covariance={cfg.predictive_covariance} residual={cfg.attention_residual} "
        f"norm={cfg.norm_type_block}/{cfg.norm_type_final} affine={cfg.layernorm_affine} "
        f"mlp={cfg.block_mlp_mode if cfg.use_block_mlp else 'none'}x{cfg.block_mlp_expansion} "
        f"decode={cfg.decode_mode} batch={cfg.batch_size}x{cfg.max_seq_len} sigma_wd={cfg.sigma_weight_decay}",
        f"  pinned (WIP-flipped defaults): {base.pinned}",
        f"  dropped retired fields: {sorted(base.dropped)}",
        f"  fields filled by current defaults: {base.defaulted}",
        f"  recorded n_params={base.recorded_params:,}" if base.recorded_params else "  recorded n_params=-",
        f"  measured throughput: {base.tokens_per_s:,.0f} tok/s ({base.throughput_source})"
        if base.tokens_per_s else "  measured throughput: unavailable",
        f"  budget={budget}  seeds (seed, data_seed)={plans[0].seeds if plans else []}",
        "-" * 118,
        f"{'arm':56s} {'status':12s} {'total':>8s} {'train':>8s} {'context':>8s} {'steps':>7s} "
        f"{'tokens':>8s} {'est h':>6s} {'disk':>7s} {'state':>7s}",
    ]
    total_hours = 0.0
    total_disk = 0.0
    for plan in plans:
        params = plan.params or {}
        steps = getattr(plan.cfg, "max_steps", None) if plan.cfg is not None else None
        # Static training state: fp32 parameters, gradients and two AdamW moments of every trainable
        # parameter, the frozen tables once. Disk: a training checkpoint of that state plus
        # best_model.pt and final_model.pt. Estimates, activations excluded.
        state = ((16 * params["trainable"] + 4 * (params["stored"] - params["trainable"])) / 1e9
                 if params.get("stored") else None)
        disk = (state + 8 * params["stored"] / 1e9) if state is not None else None
        hours = plan.est_hours
        if hours is not None and plan.status in ("ready", "deferred"):
            total_hours += hours * (len(plan.seeds) - len(plan.reused))
        if disk is not None and plan.status in ("ready", "deferred"):
            total_disk += disk * len(plan.seeds)
        lines.append(
            f"{plan.label[:56]:56s} {plan.status:12s} {_fmt_count(params.get('total')):>8s} "
            f"{_fmt_count(params.get('trainable')):>8s} {_fmt_count(params.get('context_processing')):>8s} "
            f"{steps if steps is not None else '-':>7} {_fmt_count(plan.tokens):>8s} "
            f"{f'{hours:.2f}' if hours is not None else '-':>6s} "
            f"{f'{disk:.1f}GB' if disk is not None else '-':>7s} {f'{state:.1f}GB' if state is not None else '-':>7s}")
        if plan.diff:
            shown = ", ".join(f"{key}: {a!r}->{b!r}" for key, (a, b) in plan.diff.items())
            lines.append(f"    changes: {shown}")
        if plan.dataset != base.dataset:
            lines.append(f"    dataset: {plan.dataset}")
        if plan.reason:
            lines.append(f"    {plan.status}: {plan.reason}")
        lines.extend(f"    note: {note}" for note in plan.notes)
    lines.append("-" * 118)
    lines.append(f"estimated VFE GPU-hours for ready arms x seeds at the base throughput: {total_hours:.1f} "
                 "(baseline arms not estimated; per-arm cost differs from the base)")
    lines.append(f"estimated disk for ready arms x seeds: {total_disk:.0f} GB (the compute-matched arm at the "
                 "matched width)")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------
# Runs on disk and the manifest
# ----------------------------------------------------------------------------------------------

def run_label(slug: str, seed: int, data_seed: int) -> str:
    return f"{slug}_s{seed}_d{data_seed}"


def is_complete(run_dir: Path) -> bool:
    r"""Whether ``run_dir`` holds a complete ``summary.json`` (a finite ``test_ce``)."""
    try:
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    value = summary.get("test_ce") if isinstance(summary, dict) else None
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def find_complete_run(arm_dir: Path, label: str) -> Optional[Path]:
    r"""The complete run of ``label`` in ``arm_dir`` (``<prefix>_<label>`` or ``..._<label>_<n>``)."""
    if not arm_dir.is_dir():
        return None
    pattern = re.compile(rf"^.+_{re.escape(label)}(_\d+)?$")
    for child in sorted(arm_dir.iterdir()):
        if child.is_dir() and pattern.match(child.name) and is_complete(child):
            return child
    return None


def completed_run(
    exp_dir:   Path,
    label:     str,
    seed:      int,
    data_seed: int,

    *,
    base:      Optional[Base] = None,
    budget:    Optional[str] = None,
    arms:      Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Optional[Path]:
    r"""The complete run of arm ``label`` at ``(seed, data_seed)``, or None.

    A run in the experiment directory wins. Otherwise, under the ``base_protocol`` budget, the
    evaluated arm at the base run's own ``(seed, data_seed)`` is the base run itself when it is
    complete: the same configuration, budget, and seeds.
    """
    slug = (ARMS if arms is None else arms)[label]["slug"]
    found = find_complete_run(exp_dir / slug, run_label(slug, seed, data_seed))
    if found is not None:
        return found
    if (base is not None and budget == "base_protocol" and label == EVALUATED
            and base.data_seed is not None and (int(base.cfg.seed), base.data_seed) == (seed, data_seed)
            and is_complete(base.run_dir)):
        return base.run_dir
    return None


def experiment_dir(out_root: Union[str, Path], base: str, budget: str) -> Path:
    root = Path(out_root)
    return (root if root.is_absolute() else ROOT / root) / base / budget


def build_manifest(
    exp_dir: Path,
    plans:   Sequence[ArmPlan],

    *,
    base:    Base,
    budget:  str,
    seeds:   Sequence[Tuple[int, int]],
    arms:    Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    r"""The manifest :mod:`vfe3.paper` reads: every table row, with the complete runs of the plan's seeds."""
    arms = ARMS if arms is None else arms
    by_label = {plan.label: plan for plan in plans}
    manifest_arms: Dict[str, Any] = {}
    rows: Dict[str, Any] = {}
    evaluated_run = None
    for label, spec in arms.items():
        runs = []
        for seed, data_seed in seeds:
            found = completed_run(exp_dir, label, seed, data_seed, base=base, budget=budget, arms=arms)
            if found is not None:
                runs.append(str(found))
        manifest_arms[label] = {"group": spec["group"], "runs": runs}
        plan = by_label.get(label)
        rows[label] = {"status": plan.status if plan else "not selected",
                       "reason": plan.reason if plan else None,
                       "changes": {k: [repr(a), repr(b)] for k, (a, b) in plan.diff.items()} if plan else {},
                       "complete": len(runs)}
        if label == EVALUATED and runs:
            evaluated_run = runs[0]
    return {
        "arms": manifest_arms,
        "evaluated_run": evaluated_run,
        "selection": "best",
        "out_dir": str(exp_dir / "paper_artifacts"),
        "paper_arms": {"base": base.name, "base_run": str(base.run_dir), "budget": budget,
                       "seeds": [list(pair) for pair in seeds], "rows": rows,
                       "dropped_config_fields": sorted(base.dropped), "pinned": base.pinned},
    }


def write_manifest(exp_dir: Path, manifest: Mapping[str, Any]) -> Path:
    exp_dir.mkdir(parents=True, exist_ok=True)
    path = exp_dir / "m2_manifest.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    tmp.replace(path)
    return path


# ----------------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------------

def ladder_widths(cfg: Any, multiples: Sequence[float]) -> List[int]:
    r"""Widths of the pre-declared ladder: ``multiples`` of the matched width, rounded to whole heads."""
    d = int(cfg.d_head)
    return sorted({max(d, int(round(cfg.embed_dim * float(m) / d)) * d) for m in multiples})


def _find_calibration(calibration_dir: Path, width: int, seed: int, steps: int) -> Optional[Path]:
    r"""A kept calibration run of ``width`` and ``seed`` over ``steps`` steps with a readable throughput."""
    if not calibration_dir.is_dir():
        return None
    api = _baseline_api()
    pattern = re.compile(rf"^calibration_K{int(width)}_s{int(seed)}(_\d+)?$")
    for child in sorted(calibration_dir.iterdir()):
        if not (child.is_dir() and pattern.match(child.name)):
            continue
        try:
            record = json.loads((child / "config.json").read_text(encoding="utf-8"))
            recorded = record.get("config", record)
            if recorded.get("max_steps") != int(steps) or recorded.get("embed_dim") != int(width):
                continue
            api.calibrated_tokens_per_s(child)
        except (OSError, ValueError, AttributeError):
            continue
        return child
    return None


def compute_matched_config(
    plan:              ArmPlan,
    exp_dir:           Path,

    *,
    calibration_steps: int,
    width_ladder:      Sequence[float],
    device:            Optional[str] = None,
    logger:            logging.Logger,
    base:              Optional[Base] = None,
    budget:            Optional[str] = None,
) -> Optional[Any]:
    r"""The compute-matched baseline config, or None while the evaluated arm has no complete run.

    Target :math:`h`: the mean measured train-only device-hours
    (:func:`vfe3.baseline.measured_train_device_hours`) of the evaluated arm's complete runs of the
    plan's seeds. Every ladder width gets a training-only calibration run of ``calibration_steps``
    steps (``train_baseline.calibrate_widths``; kept runs of the same width and length are reused),
    whose median train-step throughput (:func:`vfe3.baseline.calibrated_tokens_per_s`) feeds the
    pre-declared :func:`vfe3.baseline.width_ladder_rule`; it returns the width and
    ``max_steps = floor(3600 h tau / tokens_per_step)``. The evaluation cadence follows the budget
    rule (``max_steps // 20``, one training checkpoint at the end). The rule's record is written to
    ``<arm dir>/width_rule.json``.
    """
    from train_baseline import calibrate_widths
    from vfe3.data.datasets import cached_token_count

    api = _baseline_api()
    runs = [completed_run(exp_dir, EVALUATED, s, d, base=base, budget=budget) for s, d in plan.seeds]
    runs = [run for run in runs if run is not None]
    if not runs:
        logger.warning("%s deferred: the evaluated arm has no complete run yet", plan.label)
        return None
    hours = sum(api.measured_train_device_hours(run)["train_hours"] for run in runs) / len(runs)
    seed, data_seed = plan.seeds[0]
    calibration_dir = exp_dir / "_calibration"
    widths = ladder_widths(plan.cfg, width_ladder)
    missing = [w for w in widths if _find_calibration(calibration_dir, w, seed, calibration_steps) is None]
    if missing:
        logger.info("calibrating %s at widths %s (%d steps each)", plan.label, missing, calibration_steps)
        calibrate_widths(plan.cfg, missing, dataset=plan.dataset, out_root=calibration_dir, seed=seed,
                         steps=int(calibration_steps), log_interval=max(1, int(calibration_steps) // 10),
                         data_seed=data_seed, device=device, logger=logger)
    calibrations = {w: _find_calibration(calibration_dir, w, seed, calibration_steps) for w in widths}
    absent = [w for w, run in calibrations.items() if run is None]
    if absent:
        raise RuntimeError(f"no readable calibration run for width(s) {absent} in {calibration_dir}")
    taus = {w: float(api.calibrated_tokens_per_s(run)["tokens_per_s"]) for w, run in calibrations.items()}
    rule = api.width_ladder_rule(plan.cfg, hours=hours, tokens_per_s=taus,
                                 unique_train_tokens=int(cached_token_count(plan.dataset, "train")))
    chosen = rule["config"]
    steps = int(chosen.max_steps)
    cfg = dataclasses.replace(chosen, eval_interval=max(1, steps // M3_POINTS), checkpoint_interval=steps)
    check_baseline_arm(plan.matched_cfg, cfg, declared=COMPUTE_MATCHED_FIELDS)
    record = {key: value for key, value in rule.items() if key != "config"}
    record.update(config=cfg.serialize(), evaluated_runs=[str(run) for run in runs],
                  calibration_runs={w: str(run) for w, run in calibrations.items()},
                  calibration_steps=int(calibration_steps))
    arm_dir = exp_dir / plan.slug
    arm_dir.mkdir(parents=True, exist_ok=True)
    (arm_dir / "width_rule.json").write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    logger.info("%s: %.3f train device-hours -> width %d, max_steps %d (pass cap binds: %s)",
                plan.label, hours, cfg.embed_dim, steps, rule["cap_binds"])
    return cfg


def run_plan(
    base:     Base,
    plans:    Sequence[ArmPlan],

    *,
    budget:   str,
    out_root: Union[str, Path],
    device:   Optional[str] = None,
    calibration_steps: int = 200,
    width_ladder: Sequence[float] = (1, 1.5, 2, 3, 4, 6, 8),
    logger:   Optional[logging.Logger] = None,
    arms:     Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Path:
    r"""Train every ready (arm, seed) pair of ``plans`` without a complete run; return the manifest path.

    The manifest is rewritten before the first run and after every run, so an interrupted launch
    leaves a manifest of the runs that completed.
    """
    from vfe3.experiment import run_vfe

    logger = logger or logging.getLogger("paper_arms")
    invalid = [plan for plan in plans if plan.status == "invalid"]
    if invalid:
        raise SystemExit("refusing to train while arms are invalid:\n  "
                         + "\n  ".join(f"{plan.label}: {plan.reason}" for plan in invalid))
    exp_dir = experiment_dir(out_root, base.name, budget)
    seeds = plans[0].seeds if plans else []

    def refresh() -> Path:
        return write_manifest(exp_dir, build_manifest(exp_dir, plans, base=base, budget=budget,
                                                      seeds=seeds, arms=arms))

    refresh()
    for plan in plans:
        if plan.status not in ("ready", "deferred"):
            logger.info("skip %s (%s: %s)", plan.label, plan.status, plan.reason)
            continue
        cfg = plan.cfg
        if plan.status == "deferred":
            cfg = compute_matched_config(plan, exp_dir, calibration_steps=calibration_steps,
                                         width_ladder=width_ladder, device=device, logger=logger,
                                         base=base, budget=budget)
            if cfg is None:
                continue
        arm_dir = exp_dir / plan.slug
        for seed, data_seed in plan.seeds:
            label = run_label(plan.slug, seed, data_seed)
            done = completed_run(exp_dir, plan.label, seed, data_seed, base=base, budget=budget, arms=arms)
            if done is not None:
                logger.info("complete, skipped: %s", done)
                continue
            logger.info("training %s | %s | seed=%d data_seed=%d", plan.label, plan.dataset, seed, data_seed)
            if plan.kind == "vfe":
                run_vfe(cfg, dataset=plan.dataset, out_root=arm_dir, seed=seed, data_seed=data_seed,
                        label=label, device=device, logger=logger)
            else:
                _run_baseline_api()(cfg, dataset=plan.dataset, out_root=arm_dir, seed=seed,
                                    data_seed=data_seed, label=label, device=device, logger=logger)
            refresh()
    return refresh()


def main(config: Optional[Mapping[str, Any]] = None, bases: Optional[Mapping[str, str]] = None) -> int:
    r"""Click-to-run entry: plan (and unless ``dry_run``, train) every selected base."""
    config = dict(CONFIG if config is None else config)
    bases = BASES if bases is None else bases
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("paper_arms")
    budget = config["budget"]
    if budget not in BUDGETS:
        raise SystemExit(f"budget {budget!r} is not one of {sorted(BUDGETS)}")
    seeds = seed_pairs(config["seeds"])
    labels = config.get("arms")
    if isinstance(labels, str):
        tiers = {"TIER_A": TIER_A, "TIER_B": TIER_B}
        if labels not in tiers:
            raise SystemExit(f"arms {labels!r} is not None, a list of labels, or one of {sorted(tiers)}")
        labels = list(tiers[labels])
    status = 0
    for name in config["bases"]:
        started = time.perf_counter()
        base = load_base(name, bases[name])
        plans = build_plan(base, budget=budget, seeds=seeds, labels=labels,
                           baseline_overrides=config.get("baseline_overrides"))
        print(format_plan(base, plans, budget=budget), flush=True)
        print(f"(resolved in {time.perf_counter() - started:.1f}s)", flush=True)
        if any(plan.status == "invalid" for plan in plans):
            status = 1
        if not config.get("dry_run", True):
            path = run_plan(base, plans, budget=budget, out_root=config["out_root"], device=config.get("device"),
                            calibration_steps=int(config.get("calibration_steps", 200)),
                            width_ladder=tuple(config.get("width_ladder", (1, 1.5, 2, 3, 4, 6, 8))),
                            logger=logger)
            print(f"manifest: {path}", flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
