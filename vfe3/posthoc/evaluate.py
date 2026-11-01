r"""Per-target evaluation of a trained run under its own protocol, and the article-level bootstrap.

:func:`nll_pass` runs the trained predictor over evaluation windows and keeps, for every scored
target, its negative log-likelihood :math:`\ell_t = -\log p(x_t \mid x_{<t})`, the stream position
:math:`t` of the target, the target token, the window and the context length. The corpus
cross-entropy is the token-weighted mean :math:`\mathrm{CE} = \sum_t \ell_t / n`, which is exactly
what :func:`vfe3.train.evaluate` aggregates; :func:`protocol_evaluate` runs that function on the same
windows so every headline number is also produced by the run's own evaluation code, and the two are
compared.

Each batch is one forward on the replay ladder that :func:`vfe3.train.evaluate` uses (:func:`ladder_call`): it runs
inside :func:`vfe3.contracts.deferred_host_checks`, and a batch that raises at a reduced ``fp32_matmul_precision``
is replayed one rung stricter and counted in :attr:`TargetNLL.precision_replays`. Only the scored rows are decoded
(:func:`scored_target_nll`), in tiles of at most :data:`NLL_DECODE_TILE_BYTES` of logits, and a position the decoder
cannot score is decided on the covariance the decoder reads, as in the fused cross-entropy of ``evaluate``.

Intervals come from a (paired) bootstrap over resampling units :math:`a` (articles): with unit totals
:math:`S_a = \sum_{t \in a} \ell_t`, counts :math:`n_a`, and multinomial resampling weights
:math:`w_a`,

.. math::
    \mathrm{CE}^{*} = \frac{\sum_a w_a S_a}{\sum_a w_a n_a}, \qquad
    \Delta\mathrm{CE}^{*} = \frac{\sum_a w_a (S_a^{\mathrm{arm}} - S_a^{\mathrm{base}})}{\sum_a w_a n_a},

and the 95% interval is the 2.5 and 97.5 percentiles over resamples. Every arm uses the same weights
(the same seed), so the intervals of different rows are paired.
"""

from __future__ import annotations

import dataclasses
import math
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

BOOTSTRAP_RESAMPLES: int = 10_000          # resamples of the unit bootstrap
BOOTSTRAP_SEED:      int = 0               # one seed for every arm, so the rows are paired
BOOTSTRAP_CHUNK:     int = 1_000           # resamples drawn per chunk (bounds the weight matrix)
PRECISIONS: Tuple[str, ...] = ("fp32", "run")
# |CE(per-target pass) - CE(vfe3.train.evaluate)| above this (nats) marks the pass as inconsistent;
# both see the same windows and forward, so only the fused-vs-dense decode arithmetic differs.
CROSS_CHECK_TOL: float = 1e-4
# Bytes of the (rows, V) logits and their log-softmax decoded at once by scored_target_nll (one tile of scored rows).
NLL_DECODE_TILE_BYTES: int = 256 * 2**20
# Pseudo-checkpoint of load_model: the weight-EMA shadow of the latest checkpoints/step_*.pt (weight_ema_decay).
EMA_CHECKPOINT: str = "ema_weights"


def resolve_device(name: str) -> torch.device:
    r"""``"auto"`` resolves to CUDA when available, else CPU; any other name is passed to torch."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.device_count() > 0 else "cpu")
    return torch.device(name)


# ----------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------

@dataclass
class LoadedModel:
    r"""A trained model built from a checkpoint bundle, in eval mode on ``device``.

    ``weights`` is ``"raw"`` for a model bundle and ``"ema"`` for :data:`EMA_CHECKPOINT`; ``source`` is the file the
    weights were read from, and ``ema`` the weight-EMA record of an ``"ema"`` load (the step checkpoint's ``step``,
    ``updates``, ``initialized_step``, ``decay`` and ``update_interval``).
    """

    model:      torch.nn.Module = field(repr=False)
    cfg:        Any = field(repr=False)
    checkpoint: str
    precision:  str
    amp_dtype:  Optional[str]
    notes:      List[str] = field(default_factory=list)
    dropped_config_fields: Dict[str, Any] = field(default_factory=dict)
    weights:    str = "raw"
    source:     Optional[str] = None
    ema:        Optional[Dict[str, Any]] = None


def latest_training_checkpoint(run_dir: "str | Path") -> Optional[Path]:
    r"""The training checkpoint ``<run_dir>/checkpoints/step_<n>.pt`` with the largest ``n``, or None."""
    directory = Path(run_dir) / "checkpoints"
    if not directory.is_dir():
        return None
    found = []
    for path in directory.glob("step_*.pt"):
        try:
            found.append((int(path.stem.split("_", 1)[1]), path))
        except ValueError:
            continue
    return max(found)[1] if found else None


def _ema_record(bundle: Any, path: Path) -> Dict[str, Any]:
    r"""The started ``"weight_ema"`` record of a training checkpoint (:meth:`vfe3.weight_ema.WeightEMA.state_dict`)."""
    record = bundle.get("weight_ema") if isinstance(bundle, dict) else None
    if not isinstance(record, dict) or record.get("shadow") is None:
        raise ValueError(f"{path} carries no weight-EMA shadow (weight_ema_decay unset, or the EMA had not started "
                         "by this checkpoint's step)")
    names, shadow = list(record.get("names") or ()), list(record["shadow"])
    if not names or len(names) != len(shadow):
        raise ValueError(f"{path}: the weight-EMA record names {len(names)} parameters for {len(shadow)} shadows")
    return record


def _apply_ema_shadow(model: torch.nn.Module, record: Mapping[str, Any], path: Path) -> None:
    r"""Copy the shadow into the averaged parameters; every other parameter and buffer keeps its raw value.

    The swap :meth:`vfe3.weight_ema.WeightEMA.applied` makes at the end of training, by parameter name, so a
    parameter shared across layers (one name in ``named_parameters``) is written once.
    """
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in zip(record["names"], record["shadow"]):
            target = params.get(name)
            if target is None or not isinstance(value, torch.Tensor) or value.shape != target.shape \
                    or value.dtype != target.dtype:
                raise ValueError(f"{path}: the weight-EMA shadow of {name!r} does not match this model's parameter")
            target.copy_(value)


def load_model(
    run_dir:    "str | Path",
    checkpoint: str,

    *,
    device:    torch.device,
    precision: str = "fp32",
    drop_unknown_config_fields: bool = False,
) -> LoadedModel:
    r"""Build the model of ``<run_dir>/<checkpoint>`` and load its state strictly (either family).

    ``precision="fp32"`` replays in float32 with autocast off and ``fp32_matmul_precision='highest'``
    (IEEE fp32 matmuls, no TF32). ``precision="run"`` keeps the run's
    ``amp_dtype`` on CUDA (the arithmetic the run was evaluated in); on the CPU it falls back to
    float32, recorded in ``notes``. ``drop_unknown_config_fields`` ignores config fields this build
    no longer declares (see :func:`vfe3.posthoc.streams.config_from_bundle`); they are returned in
    ``dropped_config_fields``. A bundle whose config carries ``model_family =
    'conventional_transformer'`` builds :class:`vfe3.baseline.BaselineTransformer`.

    ``checkpoint=EMA_CHECKPOINT`` loads the weight-EMA weights (``weight_ema_decay``; audit 2026-10-31 A340): the
    latest ``checkpoints/step_*.pt``, with its ``"weight_ema"`` shadow copied into the averaged parameters exactly as
    :meth:`vfe3.weight_ema.WeightEMA.applied` swaps it in at the end of training. ``final_model.pt`` and
    ``best_model.pt`` hold raw weights, so the shadow exists only in the training checkpoints; it is the one finalize
    evaluated (``val_ema_ce``, ``test_ema_ce``) only when the checkpoint's EMA update count is the summary's.
    """
    from vfe3.contracts import is_vfe_config
    from vfe3.model.model import VFEModel
    from vfe3.posthoc.streams import config_from_bundle

    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    ema = checkpoint == EMA_CHECKPOINT
    if ema:
        latest = latest_training_checkpoint(run_dir)
        if latest is None:
            raise FileNotFoundError(f"{Path(run_dir) / 'checkpoints'} holds no step_*.pt training checkpoint, the "
                                    "only artifact that carries the weight-EMA shadow")
        path = latest
    else:
        path = Path(run_dir) / checkpoint
    bundle = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    record = _ema_record(bundle, path) if ema else None
    cfg, dropped = config_from_bundle(bundle["config"], source=str(path),
                                      drop_unknown_config_fields=drop_unknown_config_fields)
    notes: List[str] = []
    amp = cfg.amp_dtype if (precision == "run" and device.type == "cuda") else None
    if precision == "run" and device.type != "cuda" and cfg.amp_dtype is not None:
        notes.append(f"precision='run' on {device.type}: replayed in float32 (trained with "
                     f"amp_dtype={cfg.amp_dtype!r})")
    elif amp is None and cfg.amp_dtype is not None:
        notes.append(f"replayed in float32 with autocast off (trained with amp_dtype={cfg.amp_dtype!r})")
    # An fp32 replay is IEEE fp32: the model constructor pins the process matmul policy from its
    # config, so the run's reduced policy (TF32 under 'high') is replaced by 'highest' here.
    matmul = "highest" if precision == "fp32" else cfg.fp32_matmul_precision
    if matmul != cfg.fp32_matmul_precision:
        notes.append(f"replayed with fp32_matmul_precision='highest' (trained with "
                     f"fp32_matmul_precision={cfg.fp32_matmul_precision!r})")
    if not is_vfe_config(cfg):
        from vfe3.baseline import BaselineTransformer

        eval_cfg = dataclasses.replace(cfg, amp_dtype=amp, fp32_matmul_precision=matmul)
        model_class: Callable[[Any], torch.nn.Module] = BaselineTransformer
    else:
        eval_cfg = dataclasses.replace(cfg, amp_dtype=amp, fp32_matmul_precision=matmul)
        model_class = VFEModel
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = model_class(eval_cfg)
    notes.extend(sorted({f"model construction warning: {w.message}" for w in caught}))
    model.load_state_dict(bundle["model_state"], strict=True)
    ema_info: Optional[Dict[str, Any]] = None
    if record is not None:
        _apply_ema_shadow(model, record, path)
        ema_info = {"step": bundle.get("step"), **{key: record.get(key) for key in (
            "updates", "initialized_step", "decay", "update_interval")}}
        notes.append(f"weight-EMA shadow of {path.name} (step {ema_info['step']}, {ema_info['updates']} updates "
                     f"since step {ema_info['initialized_step']}, decay {ema_info['decay']})")
    model.to(device).eval()
    del bundle
    return LoadedModel(model=model, cfg=eval_cfg, checkpoint=checkpoint, precision=precision,
                       amp_dtype=amp, notes=notes, dropped_config_fields=dropped,
                       weights="ema" if ema else "raw", source=str(path), ema=ema_info)


def alibi_slopes(model: torch.nn.Module, seq_len: int = 2) -> Optional[List[float]]:
    r"""Resolved per-head positional slopes :math:`\alpha_h = B_{h,1,1} - B_{h,1,0}` of the log-prior."""
    prior = model._attention_log_prior(max(2, int(seq_len)), next(model.parameters()).device)
    if prior is None or prior.dim() != 3 or prior.shape[-1] < 2:
        return None
    slopes = (prior[:, 1, 1] - prior[:, 1, 0]).double().cpu()
    return [float(s) for s in slopes] if bool(torch.isfinite(slopes).all()) else None


# ----------------------------------------------------------------------------------------------
# Windows
# ----------------------------------------------------------------------------------------------

def window_start(dataset: Dataset, index: int) -> int:
    r"""Stream offset of window ``index`` (``TokenWindows`` or ``ScoredTailWindows``)."""
    if hasattr(dataset, "window_start"):
        return int(dataset.window_start(index))
    return int(index) * int(dataset.stride)


def iter_batches(
    dataset:    Dataset,
    indices:    Sequence[int],
    batch_size: int,
) -> Iterator[Tuple[np.ndarray, torch.Tensor, torch.Tensor]]:
    r"""Yield ``(window_indices, inputs (B, N), targets (B, N))`` over ``indices`` in order."""
    indices = np.asarray(indices, dtype=np.int64)
    for start in range(0, indices.size, batch_size):
        chunk = indices[start:start + batch_size]
        items = [dataset[int(i)] for i in chunk]
        yield chunk, torch.stack([x for x, _ in items]), torch.stack([y for _, y in items])


# ----------------------------------------------------------------------------------------------
# Per-target NLL
# ----------------------------------------------------------------------------------------------

@dataclass
class TargetNLL:
    r"""Scored targets of one pass, in window order (aligned across passes over the same windows)."""

    nll:        np.ndarray        # (n,) float64 -log p(target)
    target_pos: np.ndarray        # (n,) int64 stream index of the target token
    token:      np.ndarray        # (n,) int64 target token id
    window:     np.ndarray        # (n,) int64 window index
    context:    np.ndarray        # (n,) int32 tokens of context (position in window + 1)
    expected:   int               # targets offered (B * N summed over batches)
    excluded:   int               # offered targets not scored (ignore mask, degenerate decode)
    seconds:    float = 0.0
    n_batches:  int   = 0         # forward batches of the pass
    precision_replays: Dict[str, int] = field(default_factory=dict)   # completed policy -> batches replayed to it

    @property
    def n(self) -> int:
        return int(self.nll.size)

    @property
    def ce(self) -> float:
        return float(self.nll.sum() / self.nll.size) if self.nll.size else float("nan")


Observer = Any      # has update(diagnostic_capture, valid_mask, meta) and finalize()


def ladder_call(
    model: torch.nn.Module,
    call:  Callable[[], Any],
    label: str = "post-hoc evaluation forward",
) -> Tuple[Any, List[str]]:
    r"""``(call(), policies)``: one post-hoc forward on the replay ladder of :func:`vfe3.train.evaluate`.

    ``call`` runs inside :func:`vfe3.contracts.deferred_host_checks`, as the forward of ``evaluate`` does: its
    raise-only certificates join the forward's next mandatory host read instead of synchronizing one at a time, and
    the scope closes (resolving anything still queued) before the caller reads a value. A call that raises at a
    reduced ``fp32_matmul_precision`` (a fail-closed certificate tripped by TF32 or bf16 GEMM rounding) is run again
    one rung stricter, ``'medium' -> 'high' -> 'highest'`` (:func:`vfe3.train._no_update_call_with_precision_retry`,
    audit 2026-10-08 P0-1); a failure that persists at ``'highest'`` is raised (audit 2026-10-31 A048, A214).
    ``policies`` lists the policy of every attempt, the last the one that completed: more than one entry means the
    call was replayed, so its arithmetic is no longer the run's own. ``call`` must not read device values on the
    host, and it must build any capture afresh, since a failed attempt's partial outputs are discarded.
    """
    from vfe3.contracts import deferred_host_checks
    from vfe3.train import _no_update_call_with_precision_retry

    policies: List[str] = []

    def attempt() -> Any:
        policies.append(torch.get_float32_matmul_precision())
        with deferred_host_checks():
            return call()

    return _no_update_call_with_precision_retry(model, attempt, label), policies


def _target_nll(
    logits:  torch.Tensor,                     # (S, V) logits of the scored rows
    targets: torch.Tensor,                     # (S,) their target ids
) -> torch.Tensor:                             # (S,) -log p(target), at least float32
    return F.cross_entropy(logits.to(torch.promote_types(logits.dtype, torch.float32)), targets, reduction="none")


def _row_mask(flat: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    r"""The ``shape`` boolean mask that is true exactly at the row-major indices ``flat``."""
    mask = torch.zeros(shape.numel(), dtype=torch.bool, device=flat.device)
    mask[flat] = True
    return mask.view(shape)


def scored_target_nll(
    model:   torch.nn.Module,
    tokens:  torch.Tensor,                     # (B, N) input token ids on the model's device
    targets: torch.Tensor,                     # (B, N) next-token ids on the same device (-100 = ignore)

    *,
    capture: Optional[Dict[str, Any]] = None,  # forward_beliefs capture (observers); None runs captureless
) -> Tuple[torch.Tensor, torch.Tensor]:        # (valid (B, N) bool, nll (S,) of the valid rows in row-major order)
    r"""The NLL :math:`-\log p(x_t \mid x_{<t})` of every scored target of one batch, decoding only those rows.

    ``valid`` is ``targets != -100`` less the positions the decoder cannot score, decided as the fused
    cross-entropy of :func:`vfe3.train.evaluate` decides them: by
    :meth:`vfe3.model.prior_bank.PriorBank.decode_degenerate_positions` on the covariance the decoder reads
    (``VFEModel._decode_covariance``: under ``decode_covariance='encoded'`` or ``'constant'`` the stack-input
    covariance or :math:`\sigma_{\mathrm{init}}I`, not the stack output's; audit 2026-10-31 A094). A VFE model's belief
    is then decoded through ``VFEModel._decode_belief_with_context`` on the valid rows only (``decode_rows``), in
    tiles of at most :data:`NLL_DECODE_TILE_BYTES` of logits and their log-softmax, so an overlap prefix that the
    sliding-window protocol does not score costs no vocabulary readout (audit 2026-10-31 A210, prior EA5); the seam is
    the one ``forward_beliefs(..., return_logits=True)`` decodes through, so a patch of it (M5 row 12) applies here.
    A model without that seam (the conventional baseline) decodes every position and scores the valid rows.
    """
    from vfe3.contracts import is_vfe_model

    V = int(model.cfg.vocab_size)
    valid = targets != -100                                                      # (B, N)
    if not is_vfe_model(model):
        _, logits = model.forward_beliefs(tokens, return_logits=True, capture=capture)   # (B, N, V)
        return valid, _target_nll(logits[valid], targets[valid])
    frame_out: Dict[str, Any] = {}
    encoded_out: Dict[str, Any] = {}
    belief, _ = model.forward_beliefs(tokens, return_logits=False, capture=capture,
                                      _decode_frame_out=frame_out, _encoded_sigma_out=encoded_out)
    frame, encoded = frame_out.get("frame"), encoded_out.get("sigma")
    if model.cfg.decode_mode != "linear":
        sigma_decode = model._decode_covariance(belief.mu, belief.sigma, encoded, frame)   # what the decoder reads
        degenerate = model.prior_bank.decode_degenerate_positions(
            sigma_decode.to(torch.promote_types(sigma_decode.dtype, torch.float32)), canonical_frame=frame)
        if degenerate is not None:
            valid = valid & ~degenerate                                          # (B, N)
    flat = torch.nonzero(valid.reshape(-1)).squeeze(-1)                          # (S,) scored rows, row-major
    width = max(4, belief.mu.element_size())
    per_tile = max(1, NLL_DECODE_TILE_BYTES // (2 * V * width))
    parts: List[torch.Tensor] = []
    for start in range(0, int(flat.numel()), per_tile):
        rows = valid if flat.numel() <= per_tile else _row_mask(flat[start:start + per_tile], valid.shape)
        logits = model._decode_belief_with_context(belief.mu, belief.sigma, canonical_frame=frame,
                                                   encoded_sigma=encoded, decode_rows=rows)   # (1, s, V)
        parts.append(_target_nll(logits[0], targets[rows]))
    return valid, (torch.cat(parts) if parts else belief.mu.new_zeros(0))


def nll_pass(
    model:      torch.nn.Module,
    dataset:    Dataset,
    indices:    Sequence[int],

    *,
    batch_size: int,
    device:     torch.device,
    observers:  Sequence[Observer] = (),
    progress:   Optional[Callable[[int, int], None]] = None,
) -> TargetNLL:
    r"""Run the model over the windows ``indices`` and return every scored target's NLL.

    The NLL is the model's inference readout on the scored rows (:func:`scored_target_nll`). Each batch is one
    forward on the replay ladder of :func:`vfe3.train.evaluate` inside deferred host checks (:func:`ladder_call`); a
    batch that completed only at a stricter fp32 matmul policy is counted in ``precision_replays`` under that policy.
    Observers receive the per-layer diagnostic capture (``layer_conditional_captures``: routing
    weights ``beta_pre`` and capped costs ``score_energy_pre``; ``log_prior``) of each batch, the
    scored-target mask (B, N) and ``meta = {"windows", "starts", "input_valid"}`` (``input_valid``:
    (B, N) positions holding a real input token rather than final-window padding).
    """
    parts: Dict[str, List[np.ndarray]] = {k: [] for k in ("nll", "target_pos", "token", "window", "context")}
    expected = 0
    excluded = 0
    batches_run = 0
    replays: Dict[str, int] = {}
    started = time.perf_counter()
    indices = np.asarray(indices, dtype=np.int64)
    n_batches = math.ceil(indices.size / batch_size) if indices.size else 0
    for batch_index, (windows, inputs, targets) in enumerate(iter_batches(dataset, indices, batch_size)):
        B, N = targets.shape
        tokens = inputs.to(device)
        tgt = targets.to(device)

        def attempt() -> Tuple[Optional[Dict[str, Any]], torch.Tensor, torch.Tensor]:
            capture = {"diagnostic": {}} if observers else None    # afresh: a failed attempt's capture is dropped
            valid, nll = scored_target_nll(model, tokens, tgt, capture=capture)
            return capture, valid, nll

        with torch.no_grad():
            (capture, valid, nll), policies = ladder_call(model, attempt)
        if len(policies) > 1:
            replays[policies[-1]] = replays.get(policies[-1], 0) + 1
        batches_run += 1
        starts = np.array([window_start(dataset, int(w)) for w in windows], dtype=np.int64)
        if observers:
            stream = getattr(dataset, "tokens", None)
            input_valid = (torch.ones_like(valid) if stream is None else
                           torch.as_tensor(starts[:, None] + np.arange(N)[None, :] < int(stream.numel()) - 1,
                                           device=valid.device))                     # (B, N) real input tokens
            # (the last stream token is never an input: it is only the final target)
            meta = {"windows": windows, "starts": starts, "input_valid": input_valid}
            with torch.no_grad():
                for observer in observers:
                    observer.update(capture["diagnostic"], valid, meta)
        expected += B * N
        mask = valid.cpu().numpy()                                                # (B, N)
        excluded += int(B * N - mask.sum())
        rows, cols = np.nonzero(mask)                                             # row-major, as nll is ordered
        parts["nll"].append(nll.double().cpu().numpy())
        parts["target_pos"].append(starts[rows] + cols + 1)
        parts["token"].append(targets.numpy()[rows, cols].astype(np.int64))
        parts["window"].append(windows[rows])
        parts["context"].append((cols + 1).astype(np.int32))
        del capture, nll
        if progress is not None:
            progress(batch_index + 1, n_batches)
    arrays = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in parts.items()}
    return TargetNLL(nll=arrays["nll"].astype(np.float64), target_pos=arrays["target_pos"].astype(np.int64),
                     token=arrays["token"].astype(np.int64), window=arrays["window"].astype(np.int64),
                     context=arrays["context"].astype(np.int32), expected=expected, excluded=excluded,
                     seconds=time.perf_counter() - started, n_batches=batches_run, precision_replays=replays)


def protocol_evaluate(
    model:      torch.nn.Module,
    dataset:    Dataset,
    indices:    Sequence[int],

    *,
    batch_size:      int,
    device:          torch.device,
    tokens_per_char: Optional[float] = None,
) -> Dict[str, Any]:
    r""":func:`vfe3.train.evaluate` over the same windows, batched and ordered as the run's loaders."""
    from vfe3.train import evaluate

    indices = [int(i) for i in np.asarray(indices, dtype=np.int64)]
    subset = dataset if indices == list(range(len(dataset))) else Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, drop_last=False)
    started = time.perf_counter()
    result = dict(evaluate(model, loader, tokens_per_char=tokens_per_char, device=device))
    result["seconds"] = time.perf_counter() - started
    return result


# ----------------------------------------------------------------------------------------------
# Bootstrap
# ----------------------------------------------------------------------------------------------

def unit_totals(
    values: np.ndarray,          # (n,) per-target values
    units:  np.ndarray,          # (n,) resampling unit of each target
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""``(unit_ids, sums (U,), counts (U,))`` over the units that hold at least one target."""
    unit_ids, inverse = np.unique(np.asarray(units, dtype=np.int64), return_inverse=True)
    sums = np.bincount(inverse, weights=np.asarray(values, dtype=np.float64), minlength=unit_ids.size)
    counts = np.bincount(inverse, minlength=unit_ids.size).astype(np.float64)
    return unit_ids, sums, counts


def _weight_chunks(n_units: int, n_resamples: int, seed: int) -> Iterator[np.ndarray]:
    rng = np.random.default_rng(seed)
    p = np.full(n_units, 1.0 / n_units)
    remaining = n_resamples
    while remaining > 0:
        size = min(BOOTSTRAP_CHUNK, remaining)
        yield rng.multinomial(n_units, p, size=size).astype(np.float64)       # (size, U)
        remaining -= size


def bootstrap_ratio(
    numerators:   np.ndarray,       # (U,) unit sums of the numerator
    denominators: np.ndarray,       # (U,) unit sums of the denominator (target counts)

    *,
    n_resamples: int   = BOOTSTRAP_RESAMPLES,
    seed:        int   = BOOTSTRAP_SEED,
    level:       float = 0.95,
) -> Dict[str, float]:
    r"""Percentile interval of :math:`\sum_a w_a x_a / \sum_a w_a n_a` under unit resampling."""
    numerators = np.asarray(numerators, dtype=np.float64)
    denominators = np.asarray(denominators, dtype=np.float64)
    U = numerators.size
    point = float(numerators.sum() / denominators.sum()) if denominators.sum() > 0 else float("nan")
    if U < 2 or n_resamples <= 0:
        return {"point": point, "lo": float("nan"), "hi": float("nan"), "se": float("nan"),
                "n_units": int(U), "n_resamples": 0}
    draws = []
    for weights in _weight_chunks(U, n_resamples, seed):
        draws.append((weights @ numerators) / (weights @ denominators))
    stats = np.concatenate(draws)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(stats, [alpha, 1.0 - alpha])
    return {"point": point, "lo": float(lo), "hi": float(hi), "se": float(stats.std(ddof=1)),
            "n_units": int(U), "n_resamples": int(stats.size)}


def ce_interval(
    nll:   np.ndarray,
    units: np.ndarray,

    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed:        int = BOOTSTRAP_SEED,
) -> Dict[str, float]:
    r"""Unit-bootstrap interval of the corpus cross-entropy."""
    _, sums, counts = unit_totals(nll, units)
    return bootstrap_ratio(sums, counts, n_resamples=n_resamples, seed=seed)


def paired_delta_interval(
    nll_arm:  np.ndarray,
    nll_base: np.ndarray,
    units:    np.ndarray,

    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed:        int = BOOTSTRAP_SEED,
) -> Dict[str, float]:
    r"""Paired unit-bootstrap interval of :math:`\Delta\mathrm{CE}` = arm minus baseline on the same targets."""
    if nll_arm.shape != nll_base.shape:
        raise ValueError("paired arms must score the same targets")
    _, sums, counts = unit_totals(np.asarray(nll_arm) - np.asarray(nll_base), units)
    return bootstrap_ratio(sums, counts, n_resamples=n_resamples, seed=seed)


def save_nll(path: Path, result: TargetNLL, units: np.ndarray, **extra: np.ndarray) -> Path:
    r"""Write the per-target dump (and any aligned ``extra`` arrays) as a compressed ``.npz``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, nll=result.nll, target_pos=result.target_pos, token=result.token,
                        window=result.window, context=result.context,
                        unit=np.asarray(units, dtype=np.int64), **extra)
    return path
