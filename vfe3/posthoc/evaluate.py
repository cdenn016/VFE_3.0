r"""Per-target evaluation of a trained run under its own protocol, and the article-level bootstrap.

:func:`nll_pass` runs the trained predictor over evaluation windows and keeps, for every scored
target, its negative log-likelihood :math:`\ell_t = -\log p(x_t \mid x_{<t})`, the stream position
:math:`t` of the target, the target token, the window and the context length. The corpus
cross-entropy is the token-weighted mean :math:`\mathrm{CE} = \sum_t \ell_t / n`, which is exactly
what :func:`vfe3.train.evaluate` aggregates; :func:`protocol_evaluate` runs that function on the same
windows so every headline number is also produced by the run's own evaluation code, and the two are
compared.

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
    r"""A trained model built from a checkpoint bundle, in eval mode on ``device``."""

    model:      torch.nn.Module = field(repr=False)
    cfg:        Any = field(repr=False)
    checkpoint: str
    precision:  str
    amp_dtype:  Optional[str]
    notes:      List[str] = field(default_factory=list)
    dropped_config_fields: Dict[str, Any] = field(default_factory=dict)


def load_model(
    run_dir:    "str | Path",
    checkpoint: str,

    *,
    device:    torch.device,
    precision: str = "fp32",
    drop_unknown_config_fields: bool = False,
) -> LoadedModel:
    r"""Build the model of ``<run_dir>/<checkpoint>`` and load its state strictly (either family).

    ``precision="fp32"`` replays in float32 with autocast off. ``precision="run"`` keeps the run's
    ``amp_dtype`` on CUDA (the arithmetic the run was evaluated in); on the CPU it falls back to
    float32, recorded in ``notes``. ``drop_unknown_config_fields`` ignores config fields this build
    no longer declares (see :func:`vfe3.posthoc.streams.config_from_bundle`); they are returned in
    ``dropped_config_fields``. A bundle whose config carries ``model_family =
    'conventional_transformer'`` builds :class:`vfe3.baseline.BaselineTransformer`.
    """
    from vfe3.contracts import is_vfe_config
    from vfe3.model.model import VFEModel
    from vfe3.posthoc.streams import config_from_bundle

    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    path = Path(run_dir) / checkpoint
    bundle = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    cfg, dropped = config_from_bundle(bundle["config"], source=str(path),
                                      drop_unknown_config_fields=drop_unknown_config_fields)
    notes: List[str] = []
    amp = cfg.amp_dtype if (precision == "run" and device.type == "cuda") else None
    if precision == "run" and device.type != "cuda" and cfg.amp_dtype is not None:
        notes.append(f"precision='run' on {device.type}: replayed in float32 (trained with "
                     f"amp_dtype={cfg.amp_dtype!r})")
    elif amp is None and cfg.amp_dtype is not None:
        notes.append(f"replayed in float32 with autocast off (trained with amp_dtype={cfg.amp_dtype!r})")
    if not is_vfe_config(cfg):
        from vfe3.baseline import BaselineTransformer

        eval_cfg = dataclasses.replace(cfg, amp_dtype=amp)
        model_class: Callable[[Any], torch.nn.Module] = BaselineTransformer
    else:
        eval_cfg = dataclasses.replace(
            cfg, amp_dtype=amp,
            _allow_legacy_dense_decode=bool(getattr(cfg, "_allow_legacy_dense_decode", False)))
        model_class = VFEModel
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = model_class(eval_cfg)
    notes.extend(sorted({f"model construction warning: {w.message}" for w in caught}))
    model.load_state_dict(bundle["model_state"], strict=True)
    model.to(device).eval()
    del bundle
    return LoadedModel(model=model, cfg=eval_cfg, checkpoint=checkpoint, precision=precision,
                       amp_dtype=amp, notes=notes, dropped_config_fields=dropped)


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

    @property
    def n(self) -> int:
        return int(self.nll.size)

    @property
    def ce(self) -> float:
        return float(self.nll.sum() / self.nll.size) if self.nll.size else float("nan")


Observer = Any      # has update(diagnostic_capture, valid_mask, meta) and finalize()


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

    The logits are the model's inference readout (``forward_beliefs(..., return_logits=True)``).
    Observers receive the per-layer diagnostic capture (``layer_conditional_captures``: routing
    weights ``beta_pre`` and capped costs ``score_energy_pre``; ``log_prior``) of each batch, the
    scored-target mask (B, N) and ``meta = {"windows", "starts", "input_valid"}`` (``input_valid``:
    (B, N) positions holding a real input token rather than final-window padding).
    """
    cfg = model.cfg
    V = int(cfg.vocab_size)
    need_frame = getattr(cfg, "encode_mode", None) == "canonical_content_projected"
    parts: Dict[str, List[np.ndarray]] = {k: [] for k in ("nll", "target_pos", "token", "window", "context")}
    expected = 0
    excluded = 0
    started = time.perf_counter()
    indices = np.asarray(indices, dtype=np.int64)
    n_batches = math.ceil(indices.size / batch_size) if indices.size else 0
    for batch_index, (windows, inputs, targets) in enumerate(iter_batches(dataset, indices, batch_size)):
        B, N = targets.shape
        tokens = inputs.to(device)
        tgt = targets.to(device)
        capture: Optional[Dict[str, Any]] = None
        if observers:
            capture = {"diagnostic": {}}
        elif need_frame:
            capture = {}
        with torch.no_grad():
            belief, logits = model.forward_beliefs(tokens, return_logits=True, capture=capture)
            valid = tgt != -100                                                   # (B, N)
            if cfg.decode_mode != "linear":
                degenerate = model.prior_bank.decode_degenerate_positions(
                    belief.sigma.float(),
                    canonical_frame=(capture or {}).get("canonical_frame"))
                if degenerate is not None:
                    valid = valid & ~degenerate
            nll = F.cross_entropy(logits.reshape(-1, V).float(),
                                  torch.where(valid, tgt, tgt.new_full((), -100)).reshape(-1),
                                  ignore_index=-100, reduction="none").reshape(B, N)   # (B, N)
            starts = np.array([window_start(dataset, int(w)) for w in windows], dtype=np.int64)
            if observers:
                stream = getattr(dataset, "tokens", None)
                input_valid = (torch.ones_like(valid) if stream is None else
                               torch.as_tensor(starts[:, None] + np.arange(N)[None, :] < int(stream.numel()),
                                               device=valid.device))                 # (B, N) real input tokens
                meta = {"windows": windows, "starts": starts, "input_valid": input_valid}
                for observer in observers:
                    observer.update(capture["diagnostic"], valid, meta)
        expected += B * N
        mask = valid.cpu().numpy()                                                # (B, N)
        excluded += int(B * N - mask.sum())
        rows, cols = np.nonzero(mask)
        parts["nll"].append(nll.double().cpu().numpy()[rows, cols])
        parts["target_pos"].append(starts[rows] + cols + 1)
        parts["token"].append(targets.numpy()[rows, cols].astype(np.int64))
        parts["window"].append(windows[rows])
        parts["context"].append((cols + 1).astype(np.int32))
        del logits, belief, nll
        if progress is not None:
            progress(batch_index + 1, n_batches)
    arrays = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in parts.items()}
    return TargetNLL(nll=arrays["nll"].astype(np.float64), target_pos=arrays["target_pos"].astype(np.int64),
                     token=arrays["token"].astype(np.int64), window=arrays["window"].astype(np.int64),
                     context=arrays["context"].astype(np.int32), expected=expected, excluded=excluded,
                     seconds=time.perf_counter() - started)


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
