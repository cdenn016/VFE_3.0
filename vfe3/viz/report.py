r"""Single-run publication-figure DRIVER for VFE_3.0.

The figure generators in :mod:`vfe3.viz.figures` and the model-replay runners in
:mod:`vfe3.viz.extract` are pure libraries: nothing outside the tests drove them, so the
publication figure set was never produced from a real run. This module is the missing glue.
``generate_figures(run_dir)`` rebuilds the trained model from a run directory
(``config.json`` + ``final_model.pt``, with ``best_model.pt`` for older runs), runs the extract
runners and the
:mod:`vfe3.metrics` measurements they feed, and writes the single-run figure set into
``run_dir/figures/``.

This is OFF the training hot path but IS auto-run at the end of training: ``finalize_run`` calls
it on the final training weights, and ``make_figures.py`` re-runs the same set on demand for an
already-trained run. The single ``generate_embedding_figures`` switch adds the semantic-geometry
analysis of :func:`vfe3.semantics.report.analyze_run` (``figures/semantics/REPORT.md``); the remaining
figures are always attempted. Every figure is best-effort -- a plotting / dependency / shape error is
logged and skipped so one bad figure never aborts the rest -- mirroring ``RunArtifacts._save_figures``.

The SWEEP-level figures (capacity_scaling, estep_capacity, pareto_frontier, ablation_forest,
lr_grid_heatmap) need multi-run data and belong to the ablation runner, not a single run; the
two-arm ln3_symmetry_breaking figure needs a frozen-gauge AND a learned-gauge run (an
experiment to set up). Neither is produced here.
"""

import json
import logging
import math
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import torch

from vfe3 import metrics
from vfe3.contracts import is_conditional_filtering
from vfe3.config import VFE3Config
from vfe3.path_utils import prepare_owned_output_child
from vfe3.run_artifacts import (
    _atomic_replace,
    _unique_sibling_temp,
    _unique_sibling_output_temp,
    _write_json_atomic,
    collect_estep_depth_sensitivity,
    collect_estep_character,
    collect_beta_channel_decomposition,
    collect_context_sensitivity,
    collect_phi_numerics,
)
from vfe3.viz import extract
from vfe3.viz import figures as figs
from vfe3.viz.run_loading import (
    load_best_model_state as _load_best_model_state,
    load_figure_model_state as _load_figure_model_state,
    load_run_config as _load_config,
    load_run_figure_contract as _load_run_figure_contract,
    load_run_selection_contract as _load_run_selection_contract,
)
from vfe3.viz.text import supports_english_linguistic_taxonomies


CONTROLLED_BANK_TOKENS = 16_384
_FULL_PAIR_MATRIX_WORKSETS = 8
_COORDINATE_WORKSETS = 12

# Report budgets of the semantic-geometry analysis behind ``generate_embedding_figures``: about a
# quarter of the vfe3.semantics defaults, which analyze_semantics.py runs on demand. The contextual
# forward passes stay on the CPU (the package default).
SEMANTIC_ANALYSIS_OPTIONS: Dict[str, object] = {
    "max_candidates":      5000,
    "geometry_max_points": 4000,
    "frame_ref_count":     1024,
    "contextual_tokens":   8192,
    "lens_tokens":         4096,
    "contextual_points":   2048,
    "occurrence_tokens":   32768,
}

_SINGLE_RUN_FIGURE_INVENTORY = (
    "iterate_trajectory.png",
    "belief_trajectories.png",
    "attention_structure.png",
    "per_layer_diagnostics.png",
    "gauge_equivariance.png",
    "gauge_head_specialization.png",
    "belief_spectrum.png",
    "spd_ellipses.png",
    "holonomy_curvature.png",
    "numerical_trust.png",
    "vocab_probability_heatmap.png",
    "vocab_calibration.png",
    "vocab_confusion.png",
    "decode_readout.png",
    "reliability_diagram.png",
    "sigma_stratified_error.png",
    "sigma_ce_scatter.png",
)

_ENGLISH_ONLY_SINGLE_RUN_FIGURES = frozenset({
    "vocab_confusion.png",
})


def plan_single_run_figures(
    dataset:      str,
    availability: Mapping[str, bool],
) -> tuple[str, ...]:
    """Return available single-run figure filenames in stable publication order."""
    english_taxonomies = supports_english_linguistic_taxonomies(dataset)
    return tuple(
        filename
        for filename in _SINGLE_RUN_FIGURE_INVENTORY
        if availability.get(filename.removesuffix(".png"), False)
        and (english_taxonomies or filename not in _ENGLISH_ONLY_SINGLE_RUN_FIGURES)
    )


def _build_loader(dataset: str, cfg: VFE3Config, split: str):
    r"""A stable (unshuffled) loader for ``dataset``/``split``. Raises ``FileNotFoundError`` if the
    cache is absent: the figure driver never substitutes synthetic data for a real corpus (that
    would silently drive the publication figures off a toy stream). Pass ``loader=`` to
    :func:`generate_figures` to drive a custom stream instead.

    Deliberately NOT strided by ``cfg.eval_stride``: this stream feeds diagnostic figures, not a
    reported metric, and overlapping windows would make consecutive figure batches near-duplicates
    of each other for no gain. The stride belongs to the eval CONTRACT (which transitions the
    cross-entropy averages over), not to which tokens a heatmap is drawn from."""
    from vfe3.data.datasets import make_dataloader
    return make_dataloader(dataset, split, cfg.max_seq_len, cfg.batch_size,
                           shuffle=False, drop_last=False, vocab_size=cfg.vocab_size)


def _collect_token_batches(
    loader,
    device:   torch.device,
    n_batches: int,
) -> List[torch.Tensor]:
    r"""Collect CPU-hosted token batches; consumers move only their current batch to ``device``."""
    del device                                                        # retained private-call compatibility
    out: List[torch.Tensor] = []
    for batch in loader:
        tokens = _report_batch_tokens(batch)
        out.append(tokens.detach().cpu())
        if len(out) >= n_batches:
            break
    return out


def _report_batch_tokens(batch: object) -> torch.Tensor:
    """Return one report batch's token tensor without trusting configured loader dimensions."""
    tokens = batch[0] if isinstance(batch, (tuple, list)) else batch
    if not isinstance(tokens, torch.Tensor):
        raise TypeError(
            "report loader batches must be tensors or tuple/list values beginning with a tensor"
        )
    return tokens


def _collect_batches(
    loader,

    *,
    max_tokens:    Optional[int],
    max_sequences: Optional[int],
) -> List[object]:
    r"""Materialize a population bounded by the requested actual token or sequence cap."""
    extract._validate_bank_caps(max_tokens=max_tokens, max_sequences=max_sequences)
    if max_tokens is None and max_sequences is None:
        raise ValueError("one report population cap must be provided")
    out: List[object] = []
    population = 0
    for batch in loader:
        tokens = _report_batch_tokens(batch)
        if tokens.ndim != 2 or tokens.shape[0] < 1 or tokens.shape[1] < 1:
            raise ValueError("report loader token batches must be nonempty rank-two tensors")
        if max_tokens is not None:
            remaining = int(max_tokens) - population
            if remaining <= 0:
                break
            if tokens.numel() > remaining:
                batch_rows, sequence_length = tokens.shape
                if sequence_length > remaining:
                    row_limit, sequence_limit = 1, remaining
                else:
                    row_limit = min(batch_rows, remaining // sequence_length)
                    sequence_limit = sequence_length
            else:
                row_limit, sequence_limit = tokens.shape
            population += int(row_limit) * int(sequence_limit)
        else:
            remaining = int(max_sequences) - population
            if remaining <= 0:
                break
            row_limit = min(int(tokens.shape[0]), remaining)
            sequence_limit = int(tokens.shape[1])
            population += row_limit

        if row_limit == tokens.shape[0] and sequence_limit == tokens.shape[1]:
            retained = batch
        else:
            def _slice_aligned(item: object) -> object:
                if not isinstance(item, torch.Tensor):
                    return item
                if item.ndim >= 2 and item.shape[:2] == tokens.shape[:2]:
                    return item[:row_limit, :sequence_limit].clone()
                if item.ndim >= 1 and item.shape[0] == tokens.shape[0]:
                    return item[:row_limit].clone()
                return item

            if isinstance(batch, tuple):
                retained = tuple(_slice_aligned(item) for item in batch)
            elif isinstance(batch, list):
                retained = [_slice_aligned(item) for item in batch]
            else:
                retained = _slice_aligned(batch)
        out.append(retained)
        reached_cap = population >= (
            int(max_tokens) if max_tokens is not None else int(max_sequences)
        )
        if reached_cap:
            break
    return out


def _resolve_bank_budget(
    *,
    max_tokens:    Optional[int],
    max_sequences: Optional[int],
) -> 'tuple[Optional[int], Optional[int]]':
    """Resolve the exact bank cap; actual loader shapes determine how many batches are needed."""
    extract._validate_bank_caps(max_tokens=max_tokens, max_sequences=max_sequences)
    if max_tokens is None and max_sequences is None:
        max_tokens = CONTROLLED_BANK_TOKENS
    return max_tokens, max_sequences


def _report_batch_nbytes(value: object) -> int:
    """Return logical bytes of tensors retained in one materialized loader batch."""
    if isinstance(value, torch.Tensor):
        return int(value.numel()) * int(value.element_size())
    if isinstance(value, (tuple, list)):
        return sum(_report_batch_nbytes(item) for item in value)
    if isinstance(value, Mapping):
        return sum(_report_batch_nbytes(item) for item in value.values())
    return 0


def _estimated_full_vocab_bank_bytes(
    batches:    Sequence[object],
    vocab_size: int,

    *,
    decode_dtype:                    torch.dtype   = torch.float32,
    full_pair_matrix_dim:            Optional[int] = None,
    coordinate_workspace_dim:        Optional[int] = None,
    coordinate_workspace_vocab_size: Optional[int] = None,
) -> int:
    r"""Budget retained loader tensors plus the configured dense decode workspace.

    Full-vocabulary consumers decode one materialized batch at a time. The persistent contribution
    therefore comes only from the actual loader tensors, with their actual dtypes; the transient
    contribution is the largest actual token batch times two decode-dtype worksets (logits and
    softmax probabilities). A dense full-covariance functional additionally creates pairwise
    ``(B, N, V, K, K)`` matrices during factorization and solves. Eight such worksets provide a
    conservative allocation guard. Generic diagonal-family and expected-likelihood kernels instead
    create coordinate worksets shaped ``(B, N, V_or_chunk, K)``. Twelve float32-equivalent worksets
    cover the reachable near-KL family branch that retains float32 inputs alongside float64
    intermediates; the represented dimension and actual vocabulary width are budgeted separately.
    Optimized kernels leave both workspace forms unset. No configured batch or sequence dimension
    enters this estimate.
    """
    if vocab_size < 1:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")
    if (full_pair_matrix_dim is not None
            and (type(full_pair_matrix_dim) is not int or full_pair_matrix_dim < 1)):
        raise ValueError(
            f"full_pair_matrix_dim must be a positive integer or None, got "
            f"{full_pair_matrix_dim!r}"
        )
    coordinate_values = (coordinate_workspace_dim, coordinate_workspace_vocab_size)
    if (coordinate_workspace_dim is None) != (coordinate_workspace_vocab_size is None):
        raise ValueError(
            "coordinate workspace dimension and vocabulary size must be provided together"
        )
    if any(type(value) is not int or value < 1 for value in coordinate_values if value is not None):
        raise ValueError("coordinate workspace dimension and vocabulary size must be positive integers")
    if (coordinate_workspace_vocab_size is not None
            and coordinate_workspace_vocab_size > vocab_size):
        raise ValueError("coordinate workspace vocabulary size cannot exceed the full vocabulary")
    retained_bytes = sum(_report_batch_nbytes(batch) for batch in batches)
    element_size = torch.empty((), dtype=decode_dtype).element_size()
    streamed_peak = max(
        (2 * int(_report_batch_tokens(batch).numel()) * vocab_size * element_size
         for batch in batches),
        default=0,
    )
    if full_pair_matrix_dim is not None:
        streamed_peak += max(
            (_FULL_PAIR_MATRIX_WORKSETS
             * int(_report_batch_tokens(batch).numel())
             * vocab_size
             * full_pair_matrix_dim
             * full_pair_matrix_dim
             * element_size
             for batch in batches),
            default=0,
        )
    if coordinate_workspace_dim is not None and coordinate_workspace_vocab_size is not None:
        streamed_peak += max(
            (_COORDINATE_WORKSETS
             * int(_report_batch_tokens(batch).numel())
             * coordinate_workspace_vocab_size
             * coordinate_workspace_dim
             * element_size
             for batch in batches),
            default=0,
        )
    return retained_bytes + streamed_peak


def _decode_workspace_requirements(
    cfg:                VFE3Config,
    active_decode_mode: str,
) -> 'tuple[Optional[int], Optional[int], Optional[int]]':
    r"""Return dense-matrix dimension, coordinate dimension, and coordinate vocabulary width."""
    width = int(cfg.embed_dim)
    vocab_size = int(cfg.vocab_size)
    if (not cfg.diagonal_covariance
            and active_decode_mode in {"full", "family", "family_chunked"}):
        return width, None, None
    if cfg.diagonal_covariance and active_decode_mode in {"family", "family_chunked"}:
        return None, width, vocab_size
    if active_decode_mode == "expected_likelihood_chunked":
        return None, width, min(vocab_size, int(cfg.decode_chunk_size))
    return None, None, None


def _reliability_from_ce_bank(
    bank: Optional[Mapping[str, torch.Tensor]],

    *,
    n_bins: int = 15,
) -> 'Optional[List[Dict[str, float]]]':
    r"""Bin confidence and correctness from the report's current split.

    The returned schema matches :func:`vfe3.viz.figures.plot_reliability_diagram` and
    ``run_artifacts._calibration_and_strata``. Deriving it from the already-decoded CE bank keeps the
    diagram bound to this invocation's requested split and does not perform another model replay.
    """
    if bank is None:
        return None
    if type(n_bins) is not int or n_bins < 1:
        raise ValueError(f"n_bins must be a positive integer, got {n_bins!r}")
    conf = bank.get("conf")
    correct = bank.get("correct")
    if not isinstance(conf, torch.Tensor) or not isinstance(correct, torch.Tensor):
        raise ValueError("CE bank reliability requires tensor conf and correct fields")
    conf = conf.detach().reshape(-1)
    correct = correct.detach().reshape(-1).to(device=conf.device, dtype=conf.dtype)
    if conf.shape != correct.shape:
        raise ValueError("CE bank conf and correct fields must have identical shapes")
    if conf.numel() == 0:
        return None
    if not bool(torch.isfinite(conf).all()) or not bool(torch.isfinite(correct).all()):
        raise ValueError("CE bank reliability fields must be finite")
    if bool(((conf < 0.0) | (conf > 1.0)).any()):
        raise ValueError("CE bank confidence must lie in [0, 1]")
    if bool(((correct < 0.0) | (correct > 1.0)).any()):
        raise ValueError("CE bank correctness must lie in [0, 1]")

    edges = torch.linspace(0.0, 1.0, n_bins + 1, device=conf.device, dtype=conf.dtype)
    reliability: List[Dict[str, float]] = []
    for index in range(n_bins):
        mask = (conf > edges[index]) & (conf <= edges[index + 1])
        if bool(mask.any()):
            reliability.append({
                "conf": float(conf[mask].mean()),
                "acc":  float(correct[mask].mean()),
                "frac": float(mask.to(dtype=conf.dtype).mean()),
            })
    return reliability or None


def _saved_reliability_for_split(
    run_dir: Path,
    split:   str,
    logger:  logging.Logger,
) -> 'Optional[List[Dict[str, float]]]':
    r"""Load saved reliability only when its recorded split matches this report invocation."""
    path = run_dir / "research.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("top level is not a mapping")
        saved_split = payload.get("reliability_split")
        if saved_split != split:
            logger.info(
                "saved reliability ignored: research.json split %r does not match requested split %r",
                saved_split,
                split,
            )
            return None
        raw = payload.get("reliability")
        if not isinstance(raw, list):
            raise ValueError("reliability is not a list")
        reliability: List[Dict[str, float]] = []
        for entry in raw:
            if not isinstance(entry, Mapping):
                raise ValueError("reliability entry is not a mapping")
            normalized = {
                "conf": float(entry["conf"]),
                "acc":  float(entry["acc"]),
                "frac": float(entry["frac"]),
            }
            if (not all(math.isfinite(value) for value in normalized.values())
                    or any(value < 0.0 or value > 1.0 for value in normalized.values())):
                raise ValueError("reliability entry is outside [0, 1] or non-finite")
            reliability.append(normalized)
        return reliability or None
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("research.json reliability unreadable (%s); reliability figure skipped", exc)
        return None


def _prepare_missing_saved_probes(
    model:          torch.nn.Module,
    report_batches: Sequence[object],
    run_dir:        Path,
    device:         torch.device,
    logger:         logging.Logger,
    split:          str = "validation",
) -> None:
    r"""Persist missing finalization probes from the loaded model and first retained report batch."""
    depth_path = run_dir / "estep_depth_sensitivity.json"
    phi_path = run_dir / "phi_numerics.json"
    pending = []
    if is_conditional_filtering(model):
        for name in ("estep_character", "beta_channel_decomposition", "context_sensitivity"):
            path = run_dir / f"{name}.json"
            if not path.is_file() or json.loads(path.read_text(encoding="utf-8")).get("available") is False:
                pending.append(name)
    if depth_path.is_file() and phi_path.is_file() and not pending:
        return
    if not report_batches:
        logger.warning("saved probes skipped: report loader supplied no retained batch")
        return

    batch = report_batches[0]
    try:
        tokens = _report_batch_tokens(batch).to(device)
    except (RuntimeError, TypeError, ValueError) as exc:
        logger.warning("saved probes skipped: first report batch is unusable (%s)", exc)
        return

    if not phi_path.is_file():
        try:
            _write_json_atomic(phi_path, collect_phi_numerics(model, tokens))
            logger.info("saved probe -> %s", phi_path)
        except Exception as exc:
            logger.warning("phi numerical-reference probe failed (%s); skipped", exc)

    targets = batch[1] if isinstance(batch, (tuple, list)) and len(batch) >= 2 else None
    for name in pending:
        try:
            if name == "estep_character":
                record = collect_estep_character(model, tokens, depths=(1,))
            elif name == "context_sensitivity":
                record = collect_context_sensitivity(model, tokens)
            else:
                if not isinstance(targets, torch.Tensor):
                    logger.warning("beta-channel probe skipped: report batch has no tensor targets")
                    continue
                record = collect_beta_channel_decomposition(model, tokens, targets.to(device))
            record["report_replay"] = {"split": split, "n_sequences": int(tokens.shape[0]),
                                       "sequence_length": int(tokens.shape[1])}
            _write_json_atomic(run_dir / f"{name}.json", record)
            fig = getattr(figs, f"plot_{name}")(record, path=str(run_dir / f"{name}.png"))
            figs.plt.close(fig)
            logger.info("refreshed supported probe -> %s", name)
        except Exception as exc:
            logger.warning("supported probe %s failed (%s); skipped", name, exc)
    if depth_path.is_file():
        return
    if not isinstance(targets, torch.Tensor):
        logger.warning("E-step depth-sensitivity probe skipped: report batch has no tensor targets")
        return
    try:
        record = collect_estep_depth_sensitivity(
            model,
            tokens,
            targets.to(device),
            depths=(0, 1, 2, 3, 5, 8),
        )
        _write_json_atomic(depth_path, record)
        logger.info("saved probe -> %s", depth_path)
    except Exception as exc:
        logger.warning("E-step depth-sensitivity probe failed (%s); skipped", exc)


def generate_figures(
    run_dir:              'str | Path',

    *,
    split:                str                       = "validation",
    max_tokens:           Optional[int]             = None,
    max_sequences:        Optional[int]             = None,
    generate_embedding_figures: bool                = False,
    prepare_saved_probes: bool                      = False,
    checkpoint_path:      Optional[Path]            = None,
    model:                Optional[torch.nn.Module] = None,   # skip reload; drive this live model
    loader:               Optional[object]          = None,   # skip the default loader build
    device:               Optional[torch.device]    = None,
    n_e_steps:            Optional[int]             = None,
    logger:               Optional[logging.Logger]  = None,
) -> List[Path]:
    r"""Drive the model and write the single-run publication figures into ``run_dir/figures/``.

    With ``model=None`` the trained model is rebuilt from ``run_dir/config.json`` and
    ``checkpoint_path`` (default: ``final_model.pt`` when present, otherwise ``best_model.pt``).
    Pass ``model`` to drive live in-memory weights instead.
    ``loader`` defaults to a stable unshuffled loader for the run's dataset (raises if the cache is
    absent; pass ``loader`` to override).
    When both population caps are omitted, the report batches target 16,384 tokens; a loader that
    cannot supply that many (small split/dataset) drives the figures at its actual population. The
    two caps are mutually exclusive. ``generate_embedding_figures`` runs
    :func:`vfe3.semantics.report.analyze_run` on ``run_dir/best_model.pt`` with
    :data:`SEMANTIC_ANALYSIS_OPTIONS` after every other figure, writing ``figures/semantics/``; like
    the figures it is best-effort, and its ``REPORT.md`` is appended to the returned paths.
    Full-vocabulary computation is always skipped when its actual-batch,
    decode-aware workspace estimate exceeds the internal 8 GB safety guard; lighter inputs and
    figures still run. ``prepare_saved_probes`` persists missing depth-sensitivity and phi-numerics
    inputs from the already-loaded model and retained report batch so an isolated on-demand worker
    can render the corresponding finalization figures without a second report replay. ``n_e_steps``
    overrides the E-step trace length (default: the trained ``cfg.n_e_steps``). Returns the figure
    paths actually written (best-effort: a failed figure is logged and omitted).
    """
    if type(generate_embedding_figures) is not bool:
        raise TypeError("generate_embedding_figures must be bool")
    run_dir = Path(run_dir)
    figdir = prepare_owned_output_child(run_dir, "figures", role="single-run figure")
    logger = logger or logging.getLogger(__name__)

    if model is None:
        from vfe3.model.model import VFEModel
        cfg, dataset = _load_config(run_dir)
        model = VFEModel(cfg)
        if checkpoint_path is None and (run_dir / "final_model.pt").is_file():
            checkpoint_path = run_dir / "final_model.pt"
        best = checkpoint_path if checkpoint_path is not None else run_dir / "best_model.pt"
        if not best.exists():
            raise FileNotFoundError(f"no model checkpoint at {best}; train with RunArtifacts first")
        if checkpoint_path is None:
            expected_code_identity, expected_selection_data_identity = (
                _load_run_selection_contract(run_dir))
            model_state = _load_best_model_state(
                best,
                cfg,
                model.state_dict(),
                expected_code_identity,
                expected_selection_data_identity,
                map_location=device or "cpu",
            )
        else:
            expected_code_identity, expected_selection_data_identity = (
                _load_run_figure_contract(run_dir))
            model_state = _load_figure_model_state(
                best,
                cfg,
                model.state_dict(),
                expected_code_identity,
                expected_selection_data_identity,
                map_location=device or "cpu",
            )
        model.load_state_dict(
            model_state,
            strict=True,
        )
    else:
        cfg = model.cfg
        dataset = ""
        cfgj = run_dir / "config.json"
        if cfgj.exists():
            dataset = json.loads(cfgj.read_text()).get("dataset", "")

    device = device or next(model.parameters()).device
    model = model.to(device)
    model.eval()
    english_linguistic_diagnostics = supports_english_linguistic_taxonomies(dataset)
    figs.set_publication_style()

    if loader is None:
        loader = _build_loader(dataset, cfg, split)
    max_tokens, max_sequences = _resolve_bank_budget(
        max_tokens=max_tokens,
        max_sequences=max_sequences,
    )
    report_batches = _collect_batches(
        loader,
        max_tokens=max_tokens,
        max_sequences=max_sequences,
    )
    if not report_batches:
        raise RuntimeError(f"loader for {dataset!r}/{split!r} yielded no batches")
    n_batches = len(report_batches)

    active_decode_mode = cfg.decode_mode
    (full_pair_matrix_dim,
     coordinate_workspace_dim,
     coordinate_workspace_vocab_size) = _decode_workspace_requirements(
        cfg,
        active_decode_mode,
    )
    full_vocab_gb = _estimated_full_vocab_bank_bytes(
        report_batches,
        int(cfg.vocab_size),
        full_pair_matrix_dim=full_pair_matrix_dim,
        coordinate_workspace_dim=coordinate_workspace_dim,
        coordinate_workspace_vocab_size=coordinate_workspace_vocab_size,
    ) / 1e9
    skip_full_vocab = full_vocab_gb > 8.0
    if skip_full_vocab:
        logger.warning(
            "full-vocab figure inputs skipped: actual streamed decode-workspace peak estimate "
            "%.1f GB exceeds "
            "the internal 8 GB safety guard",
            full_vocab_gb,
        )

    def _safe(fn: Callable, label: str):
        r"""Run a model-replay extractor, logging+swallowing a failure so the rest proceed."""
        try:
            return fn()
        except Exception as exc:
            logger.warning("input %r failed (%s); dependent figures skipped", label, exc)
            return None

    inference_bank = _safe(lambda: extract.collect_inference_bank(
        model,
        report_batches,
        max_batches=n_batches,
        device=device,
        return_logits=False,
    ), "inference_bank")
    if inference_bank is not None:
        retained_bank_gb = extract.inference_bank_nbytes(inference_bank) / 1e9
        logger.info(
            "report inference bank retains %.3f GB across %d CPU-hosted batches",
            retained_bank_gb,
            len(inference_bank),
        )
    token_batches = (
        [record["tokens"] for record in inference_bank]
        if inference_bank is not None
        else _collect_token_batches(report_batches, device, n_batches)
    )
    tok = token_batches[0][:1].to(device)                              # one device-resident report sequence

    # ---- expensive model-replay inputs, each guarded (a failure skips only its figures) ----
    # The same-token diagnostics consume one captured forward. If capture itself fails, passing
    # snapshot=None preserves the previous per-extractor best-effort fallback instead of dropping
    # every dependent figure.
    snapshot = _safe(lambda: model.build_diagnostic_snapshot(tok), "diagnostic_snapshot")
    trace_snapshot = snapshot
    if (snapshot is not None and n_e_steps is not None
            and int(n_e_steps) != len(snapshot.trace_states) - 1):
        trace_snapshot = None
    trace       = _safe(lambda: extract.e_step_belief_trace(
        model, tok, n_iter=n_e_steps, snapshot=trace_snapshot), "e_step_belief_trace")
    layer_trace = _safe(lambda: extract.across_layer_belief_trace(
        model, tok, snapshot=snapshot), "across_layer_belief_trace")
    cstate = (
        _safe(lambda: extract.converged_state(model, tok, snapshot=snapshot), "converged_state")
        if not is_conditional_filtering(model)
        else _safe(lambda: extract.readout_state(model, tok, snapshot=snapshot), "readout_state")
    )
    decode_stats = (
        None
        if skip_full_vocab or inference_bank is None
        else _safe(
            lambda: extract.belief_ce_vocab_stats(
                model,
                inference_bank,
                device=device,
                max_batches=n_batches,
            ),
            "belief_ce_vocab_stats",
        )
    )
    ce_bank = (
        None
        if skip_full_vocab
        else (
            decode_stats[0]
            if decode_stats is not None
            else (
                _safe(lambda: extract.belief_ce_bank(
                    model, report_batches, device=device, max_batches=n_batches),
                      "belief_ce_bank")
                if inference_bank is None
                else None
            )
        )
    )   # B1/EXP-3 Sigma_q<->CE join (calibration figures)
    # score_at="entry" to match the CSV columns and the shipped attention/ figures (audit 2026-08-06
    # F19); the snapshot can only serve "converged", which is NOT the beta the forward used.
    amaps       = _safe(lambda: model.attention_maps(tok, score_at="entry"), "attention_maps")
    per_layer   = _safe(lambda: model.diagnostics_per_layer(
        tok, snapshot=snapshot), "diagnostics_per_layer")
    health      = _safe(lambda: extract.numerical_health(
        model, tok, snapshot=snapshot), "numerical_health")
    vstats = (
        None
        if skip_full_vocab
        else (
            decode_stats[1]
            if decode_stats is not None
            else (
                _safe(lambda: extract.vocab_prediction_stats(model, token_batches),
                      "vocab_prediction_stats")
                if inference_bank is None
                else None
            )
        )
    )
    readout     = _safe(lambda: extract.decode_readout(model), "decode_readout")
    run_label   = f"K{cfg.embed_dim}"

    # gpt2/cl100k decoder for the vocabulary figures' token labels and the confusion figure's category
    # bucketing (None when tiktoken is absent or the dataset has no real tokenizer -> labels by id).
    decode = None
    try:
        from vfe3.data.datasets import get_tiktoken_decoder
        decode = get_tiktoken_decoder(dataset)
    except Exception as exc:
        logger.warning("token decoder unavailable (%s); vocabulary figures label tokens by id", exc)
    if not english_linguistic_diagnostics:
        logger.info("English-only vocabulary-confusion figure disabled for dataset %r", dataset)

    # Decode-calibration reliability bins (conf/acc/frac) come from THIS invocation's already-decoded
    # CE bank. A saved fallback is accepted only when research.json records the same split; legacy
    # files with unknown provenance must not silently mix finalization/test with on-demand/validation.
    reliability = None
    if ce_bank is not None:
        try:
            reliability = _reliability_from_ce_bank(ce_bank)
        except (RuntimeError, TypeError, ValueError) as exc:
            logger.warning("current-split reliability reduction failed (%s); trying saved input", exc)
    if reliability is None:
        reliability = _saved_reliability_for_split(run_dir, split, logger)

    availability = {
        "iterate_trajectory":         trace is not None,
        "belief_trajectories":        trace is not None,
        "attention_structure":        amaps is not None,
        "per_layer_diagnostics":      per_layer is not None,
        "gauge_equivariance":         cstate is not None and "omega" in cstate
                                      and not is_conditional_filtering(model),
        "gauge_head_specialization":  cstate is not None and "exp_phi" in cstate,
        "belief_spectrum":            cstate is not None,
        "spd_ellipses":               cstate is not None,
        "holonomy_curvature":         cstate is not None and "omega" in cstate,
        "numerical_trust":            cstate is not None,
        "vocab_probability_heatmap":  vstats is not None,
        "vocab_calibration":          vstats is not None,
        "vocab_confusion":            vstats is not None and decode is not None,
        "decode_readout":             readout is not None,
        "reliability_diagram":        reliability is not None,
        "sigma_stratified_error":     ce_bank is not None,
        "sigma_ce_scatter":           ce_bank is not None,
    }
    planned_figures = frozenset(plan_single_run_figures(dataset, availability))

    written: List[Path] = []

    def _emit(name: str, thunk: Callable[[str], object], available: bool) -> None:
        r"""Write ``figures/<name>.png`` from ``thunk(path)``; skip when an input is missing."""
        if not available:
            logger.info("figure %r skipped (input unavailable)", name)
            return
        _before = set(figs.plt.get_fignums())            # registry snapshot for the leak sweep below
        try:
            path = figdir / f"{name}.png"
            with _unique_sibling_output_temp(path) as temporary_path:
                fig = thunk(str(temporary_path))
                outcomes = getattr(fig, "_vfe3_publication_outcomes", None)
                if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
                    raise RuntimeError(f"figure {name!r} did not write a nonempty temporary output")
                _atomic_replace(path, temporary_path)
            figs.plt.close(fig)
            written.append(path)
            logger.info("figure -> %s", path)
            if isinstance(outcomes, Mapping):
                sidecar = outcomes.get("sidecar")
                if isinstance(sidecar, Mapping) and sidecar.get("published"):
                    logger.info("sidecar -> %s", sidecar.get("path"))
                elif isinstance(sidecar, Mapping) and sidecar.get("error"):
                    logger.warning("figure %r sidecar failed (%s); figure retained",
                                   name, sidecar.get("error"))
        except Exception as exc:
            # The thunk can register a pyplot figure and then raise (tight_layout/savefig) before
            # _emit ever receives it; close what it registered (audit 2026-07-01 round-3).
            for num in set(figs.plt.get_fignums()) - _before:
                figs.plt.close(num)
            logger.warning("figure %r failed (%s); continuing", name, exc)

    _emit("iterate_trajectory",
          lambda p: figs.plot_iterate_trajectory(trace, diagonal=cfg.diagonal_covariance,
                                                divergence_family=cfg.divergence_family, path=p),
          "iterate_trajectory.png" in planned_figures)
    _emit("belief_trajectories",
          lambda p: figs.plot_belief_trajectories(trace, layer_trace, path=p),
          "belief_trajectories.png" in planned_figures)
    _emit("attention_structure",
          lambda p: figs.plot_attention_structure(amaps, path=p),
          "attention_structure.png" in planned_figures)
    # Per-head beta attention heatmaps are NOT emitted here: training already writes them at
    # every eval_interval (RunArtifacts.save_attention_maps under <run_dir>/attention/), so
    # end-of-training copies in figures/ were redundant.
    _emit("per_layer_diagnostics",                               # the depth axis the aggregates collapse
          lambda p: figs.plot_per_layer_diagnostics(per_layer, path=p),
          "per_layer_diagnostics.png" in planned_figures)
    _emit("gauge_equivariance",
          lambda p: figs.plot_gauge_equivariance(metrics.gauge_equivariance_residual(
              cstate["mu"], cstate["sigma"], cstate["omega"], model.group,
              kappa=cfg.kappa_beta, renyi_order=cfg.renyi_order, kl_max=cfg.kl_max, eps=cfg.eps,
              diagonal=cfg.diagonal_covariance, divergence_family=cfg.divergence_family,
              full_cov_kl_precision=model.full_cov_kl_precision), path=p),
          "gauge_equivariance.png" in planned_figures)
    _emit("gauge_head_specialization",
          lambda p: figs.plot_gauge_head_specialization(
              metrics.per_head_gauge_invariants(cstate["exp_phi"], model.group.irrep_dims),
              # cstate["exp_phi"] is the FINAL block's gauge frame, so pair it with the FINAL layer's
              # per-head entropy (amaps[-1]) -- NOT the all-layer mean, which mismatched the depths.
              head_entropy=(metrics.attention_entropy_rows(amaps[-1]).mean(dim=-1) if amaps is not None else None),
              path=p),
          "gauge_head_specialization.png" in planned_figures)
    _emit("belief_spectrum",
          lambda p: figs.plot_belief_spectrum(
              cstate["sigma"], eps=cfg.eps, sigma_max=cfg.sigma_max,
              family=cfg.family, path=p),
          "belief_spectrum.png" in planned_figures)
    _emit("spd_ellipses",
          lambda p: figs.plot_spd_ellipses(
              cstate["mu"], cstate["sigma"], eps=cfg.eps,
              family=cfg.family, path=p),
          "spd_ellipses.png" in planned_figures)
    _emit("holonomy_curvature",
          lambda p: figs.plot_holonomy_curvature(
              metrics.holonomy_deviation_sampled(cstate["omega"]),
              curvature=metrics.curvature_field(cstate["omega"]), path=p),   # Panel C: spatial curvature field
          "holonomy_curvature.png" in planned_figures)
    _emit("numerical_trust",
          lambda p: figs.plot_numerical_trust(
              (cstate["guard"] if "guard" in cstate else metrics.guard_saturation(
                  cstate["sigma"], cstate["energy"], cstate["self_div"],
                  eps=cfg.eps, sigma_max=cfg.sigma_max, kl_max=cfg.kl_max)),
              health if health is not None else {},
              causal=(metrics.causal_sanity(amaps) if amaps is not None else None), path=p),
          "numerical_trust.png" in planned_figures)
    # Next-token vocabulary-probability figures (single-arm here; the cross-run K70-vs-K120 contrast
    # is the two-arm vocab_comparison_figures driver). vocab_confusion needs the token decoder for its
    # category bucketing; decode_readout is None (skipped) on prior-bank decode paths.
    _emit("vocab_probability_heatmap",
          lambda p: figs.plot_vocab_probability_heatmap([{**vstats, "label": run_label}], decode=decode, path=p),
          "vocab_probability_heatmap.png" in planned_figures)
    _emit("vocab_calibration",
          lambda p: figs.plot_vocab_calibration([{**vstats, "label": run_label}], decode=decode, path=p),
          "vocab_calibration.png" in planned_figures)
    _emit("vocab_confusion",
          lambda p: figs.plot_vocab_confusion([{**vstats, "label": run_label}], decode=decode, path=p),
          "vocab_confusion.png" in planned_figures)
    _emit("decode_readout",
          lambda p: figs.plot_decode_readout([{**readout, "label": run_label}], decode=decode, path=p),
          "decode_readout.png" in planned_figures)

    # B1/EXP-3 -- Sigma_q-as-calibrated-uncertainty: the decode-reliability control + the two
    # Sigma_q<->CE figures (stratified error curve, rho scatter) off the belief_ce_bank join.
    _emit("reliability_diagram",
          lambda p: figs.plot_reliability_diagram(reliability, path=p),
          "reliability_diagram.png" in planned_figures)
    # Frozen-at-encode covariance: tr Sigma is the input-token table variance, relabeled as such.
    frozen_covariance = metrics.covariance_frozen_at_encode(cfg)
    _emit("sigma_stratified_error",
          lambda p: figs.plot_sigma_stratified_error(
              ce_bank, frozen_covariance=frozen_covariance, path=p),
          "sigma_stratified_error.png" in planned_figures)
    _emit("sigma_ce_scatter",
          lambda p: figs.plot_sigma_ce_scatter(
              ce_bank, frozen_covariance=frozen_covariance, path=p),
          "sigma_ce_scatter.png" in planned_figures)

    # Per-layer metrics CSV (rows = inference depth): the metrics.csv is final-block-only, so this is
    # the only place the per-layer belief-channel free energy / holonomy / gauge / belief geometry is
    # written to disk. Best-effort, like the figures.
    if per_layer is not None:
        try:
            import csv
            keys = list(per_layer)
            n_layers = len(per_layer[keys[0]]) if keys else 0
            csv_path = run_dir / "metrics_per_layer.csv"
            with _unique_sibling_temp(csv_path) as temporary_path:
                with temporary_path.open("w", newline="", encoding="utf-8") as fh:
                    w = csv.writer(fh)
                    w.writerow(["layer", *keys])
                    for li in range(n_layers):
                        values = (per_layer[key][li] for key in keys)
                        w.writerow([li, *(value if isinstance(value, str) else float(value)
                                          for value in values)])
                _atomic_replace(csv_path, temporary_path)
            logger.info("wrote per-layer metrics -> %s", csv_path)
        except Exception as exc:
            logger.warning("metrics_per_layer.csv failed (%s); continuing", exc)

    if prepare_saved_probes:
        _prepare_missing_saved_probes(model, report_batches, run_dir, device, logger, split=split)

    # Semantic geometry of the selected checkpoint (vfe3.semantics), last because it is the slowest.
    if generate_embedding_figures:
        try:
            from vfe3.semantics import report as semantics_report
            written.append(semantics_report.analyze_run(
                run_dir, out_dir=figdir / "semantics", checkpoint="best_model.pt",
                options=SEMANTIC_ANALYSIS_OPTIONS))
            logger.info("semantic analysis -> %s", written[-1])
        except Exception as exc:
            logger.warning("semantic analysis failed (%s); continuing", exc)

    logger.info("wrote %d single-run figures to %s", len(written), figdir)
    return written


def vocab_comparison_figures(
    run_dirs:  'list[str | Path]',
    out_dir:   'str | Path',

    *,
    labels:        Optional[List[str]]      = None,
    device:        Optional[torch.device]   = None,
    split:         str                      = "validation",
    max_sequences: int                      = 256,
    taxonomy:      str                      = "function_content",
    logger:        Optional[logging.Logger] = None,
) -> List[Path]:
    r"""Cross-run vocabulary-probability comparison (e.g. K70 vs K120) into ``out_dir``.

    Loads each trained run, runs the single ``vocab_prediction_stats`` pass, and writes the four
    arm-list figures (probability heatmap / calibration / confusion / decode readout) with one
    column per run -- the side-by-side collapse contrast the single-run :func:`generate_figures`
    pipeline cannot make (it sees one model). Every arm must use the same tokenizer tag; mixed
    tokenizers are rejected before any model load or plotting because one shared decoder cannot
    label them honestly. Each figure is best-effort: a failure is logged and skipped. ``labels``
    default to ``K<embed_dim>``; the decoder is taken from the first run's dataset after that identity
    check (the confusion figure is skipped when no tokenizer is available).
    """
    from vfe3.data.datasets import _tokenizer_tag
    from vfe3.model.model import VFEModel

    prepared = []
    for rd in run_dirs:
        run_dir = Path(rd)
        cfg, dataset = _load_config(run_dir)
        prepared.append((run_dir, cfg, dataset, _tokenizer_tag(dataset)))
    tokenizer_tags = {arm[3] for arm in prepared}
    if len(tokenizer_tags) > 1:
        details = ", ".join(f"{run_dir}: {tag}" for run_dir, _, _, tag in prepared)
        raise ValueError(f"mixed tokenizer tags in vocabulary comparison: {details}")
    english_linguistic_diagnostics = all(
        supports_english_linguistic_taxonomies(dataset)
        for _, _, dataset, _ in prepared
    )
    if not english_linguistic_diagnostics:
        (logger or logging.getLogger(__name__)).info(
            "English-only vocabulary confusion disabled for non-English comparison datasets"
        )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger = logger or logging.getLogger(__name__)
    figs.set_publication_style()

    arms_pred: List[dict] = []
    arms_readout: List[dict] = []
    decode = None
    for i, (rd, cfg, dataset, _) in enumerate(prepared):
        model = VFEModel(cfg)
        best = rd / "best_model.pt"
        if not best.exists():
            raise FileNotFoundError(f"no best_model.pt in {rd}; train with RunArtifacts first")
        expected_code_identity, expected_selection_data_identity = (
            _load_run_selection_contract(rd))
        model.load_state_dict(
            _load_best_model_state(
                best,
                cfg,
                model.state_dict(),
                expected_code_identity,
                expected_selection_data_identity,
                map_location=device or "cpu",
            ),
            strict=True,
        )
        dev = device or next(model.parameters()).device
        model = model.to(dev)
        model.eval()
        loader = _build_loader(dataset, cfg, split)
        n_batches = max(2, -(-max_sequences // max(cfg.batch_size, 1)))
        token_batches = _collect_token_batches(loader, dev, n_batches)
        label = labels[i] if labels and i < len(labels) else f"K{cfg.embed_dim}"
        arms_pred.append({**extract.vocab_prediction_stats(model, token_batches), "label": label})
        ro = extract.decode_readout(model)
        if ro is not None:
            arms_readout.append({**ro, "label": label})
        if decode is None:
            try:
                from vfe3.data.datasets import get_tiktoken_decoder
                decode = get_tiktoken_decoder(dataset)
            except Exception as exc:
                logger.warning("token decoder unavailable (%s); confusion figure skipped", exc)

    written: List[Path] = []

    def _emit(name: str, thunk: Callable[[], object], available: bool) -> None:
        if not available:
            logger.info("comparison figure %r skipped (input unavailable)", name)
            return
        before = set(figs.plt.get_fignums())
        fig = None
        try:
            path = out / f"{name}.png"
            fig = thunk()
            with _unique_sibling_output_temp(path) as temporary_path:
                figs._save(fig, str(temporary_path))
                if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
                    raise RuntimeError(f"comparison figure {name!r} was not written")
                _atomic_replace(path, temporary_path)
            written.append(path)
            logger.info("figure -> %s", path)
        except Exception as exc:
            logger.warning("comparison figure %r failed (%s); continuing", name, exc)
        finally:
            if fig is not None:
                figs.plt.close(fig)
            for num in set(figs.plt.get_fignums()) - before:
                figs.plt.close(num)

    _emit("vocab_probability_heatmap_compare",
          lambda: figs.plot_vocab_probability_heatmap(arms_pred, decode=decode), bool(arms_pred))
    _emit("vocab_calibration_compare",
          lambda: figs.plot_vocab_calibration(arms_pred, decode=decode), bool(arms_pred))
    _emit("vocab_confusion_compare",
          lambda: figs.plot_vocab_confusion(arms_pred, decode=decode, taxonomy=taxonomy),
          bool(arms_pred) and decode is not None and english_linguistic_diagnostics)
    _emit("decode_readout_compare",
          lambda: figs.plot_decode_readout(arms_readout, decode=decode), bool(arms_readout))

    logger.info("wrote %d comparison figures to %s", len(written), out)
    return written
