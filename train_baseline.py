r"""Click-to-run training entry for the conventional-transformer rows of Table M2.

Edit the CONFIG block below and run ``python train_baseline.py`` (no CLI). The baseline is always
derived from a declared VFE base run with :meth:`vfe3.baseline.BaselineConfig.matched_to`, never
from the WIP ``train_vfe3.config``: the base run's ``config.json`` is read with
:func:`vfe3.legacy_config.lenient_config_from_serialized` (older runs carry retired fields), and its
tokenizer size, width, heads, depth, MLP width, context, batch, stride, budget, schedule floors, clip,
precision, cadence, attention prior and unigram decode prior are copied.

Each seed runs through the shared training loop (:func:`vfe3.train.train`) and
:func:`vfe3.run_artifacts.finalize_run`, exactly as ``train_vfe3._run_once`` does for the VFE:
``metrics.csv`` (``val_ce``, ``train_sample_ce``, ``tokens_seen``, ``wall_clock_s``, throughput),
``summary.json`` (``test_ce`` at the final weights, ``test_best_*`` at the validation-selected
weights), ``resources.json`` (per-type tables against context-processing parameters, device-hours),
``provenance.json``, ``best_model.pt``, ``final_model.pt`` and checkpoints, in
``<OUT_ROOT>/<timestamp>_<label>/``, renamed ``<test_ppl>_<label>/`` after finalize. The loaders are
built with the exact arguments of ``train_vfe3._select_loader``; ``DATA_SEED=None`` sets the data
order seed to the model seed. :func:`check_base_data` refuses a launch whose dataset, tokenizer, data
seed or token cap differs from what the base run recorded.

Three optional stages precede the seeded runs:

* ``LR_SWEEP``: short runs at each peak learning rate (``LR_SWEEP_STEPS`` steps), selected on
  validation only (``best_val_ppl``); the winner replaces ``lr`` for the seeded runs.
* ``CALIBRATE_WIDTHS``: short training-only runs at each width of the ladder that record the
  training throughput in ``metrics.csv`` (the input of the compute-matched width rule).
* ``COMPUTE_MATCH``: the matched-measured-compute arm. Its budget is the measured training
  device-hours of the retrained evaluated arm (:func:`vfe3.baseline.measured_train_device_hours`);
  the width comes from the pre-declared rule :func:`vfe3.baseline.width_ladder_rule` over the
  calibrated throughputs, and ``max_steps`` spends the whole budget.
"""

import os
if os.environ.get("VFE3_ALLOW_DUPLICATE_OPENMP") == "1":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from vfe3.quiet import silence_toolchain_warnings
silence_toolchain_warnings()

import json
import logging
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from vfe3.baseline import (
    CALIBRATION_RECORD,
    BaselineConfig,
    BaselineTransformer,
    calibrated_tokens_per_s,
    calibration_mismatch,
    measured_train_device_hours,
    width_ladder_rule,
)
from vfe3.data.datasets import (
    _tokenizer_tag,
    _validate_path_component,
    cached_token_count,
    make_dataloader,
    tokens_per_char as _tokens_per_char,
)
from vfe3.runtime import seed_everything
from vfe3.train import evaluate, train


# ================================================================================================
# CONFIG (click-to-run)
# ================================================================================================

# The VFE base runs of the M2 arms (config.json is read from each directory).
BASE_PRESETS: Dict[str, str] = {
    "diag_K128": "D:/DATA_KEEP/39.77_wikitext-103_K128_n=128",
    "diag_K64":  "D:/DATA_KEEP/52.91_wikitext-103_K64_n=128",
    "full_K64":  "vfe3_runs/Full_Gaussian_mom-match_Template_routing/55.07_wikitext-103_K64_moment_matched_noAtnnRes",
    # 34.48 test PPL: K=128, 4 heads of 32, 4 layers, exponential content chart, 150,000 steps (614.4M
    # tokens), context 128, stride 32, batch 32, model seed 6, data seed 3, bf16, fp32 matmul "highest".
    "k128_exp":  "vfe3_runs/34.48_wikitext-103_K128_block_glk_linear_mlp_s6",
}
BASE = "k128_exp"                   # a preset name or a run directory
DATASET = "wikitext-103"
OUT_ROOT = "vfe3_runs/baselines"
SEEDS: Sequence[int] = (6,)
DATA_SEED: Optional[int] = 3        # the data-order seed of the base run (3 for k128_exp); None -> model seed
MAX_TOKENS: Optional[int] = None    # cap the training stream (smoke runs only)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PER_TYPE_TABLE = "none"             # "none" | "block_read" (Table M2, per-type table of the frame table's size)
# Everything in MATCHED_FIELDS is copied from the base run (K, H, L, context, batch, stride, 150,000 steps,
# warmup, LR floor, clip, bf16, "highest" matmul precision, cadence, unigram decode prior), and a base with
# pos_rotation="rope" also gives the baseline rotary query/key positions at the base's rope_base (the
# VFE's comparison-space RoPE). A base whose position is carried by learned frames (pos_phi) has a uniform
# causal prior and no RoPE, so the copy would leave the baseline with no positional signal at all (NoPE);
# such a base gets ALiBi (POSITION_FALLBACK), the standard zero-parameter choice and what the earlier 90.15
# baseline used. OVERRIDES set last and win, e.g. {"beta_attention_prior": "causal"} for the strict copy.
# Weight decay stays at the baseline default 0.1 on every matrix and table (the base run decays only
# the frames, 0.1, and the MLPs, 0.01) and is not swept.
POSITION_FALLBACK: Dict[str, Any] = {"beta_attention_prior": "causal_alibi"}   # only for a base without RoPE
OVERRIDES: Dict[str, Any] = {}      # any BaselineConfig field
LABEL: Optional[str] = None         # stable run-folder label; None -> <dataset>_K<K>_baseline[_block_read]_s<seed>

# The baseline's peak lr (6e-4) was set for a 30,000-step run; the budget here is five times longer, so
# select it on validation. Each launch sweeps its own arm (with PER_TYPE_TABLE="block_read" the read-map
# lr follows the swept lr) and trains the winner. A short schedule favors larger lr, so a winner at the
# top of the grid (3e-3) means the grid should be extended, not trusted. Set () to skip the sweep.
LR_SWEEP: Sequence[float] = (3e-4, 6e-4, 1e-3, 2e-3, 3e-3)    # validation-only selection
LR_SWEEP_STEPS = 6000

CALIBRATE_WIDTHS: Sequence[int] = ()   # e.g. (128, 256, 384, 512, 768); training-only throughput runs
CALIBRATION_STEPS = 2000
CALIBRATION_LOG_INTERVAL = 100

# Matched measured compute: None, or {"evaluated_run": <retrained evaluated arm dir>,
# "calibration_runs": {width: <calibration run dir>}, "unique_train_tokens": int | None}.
COMPUTE_MATCH: Optional[Dict[str, Any]] = None


# ================================================================================================
# Library
# ================================================================================================

def load_base_config(base: str) -> Any:
    r"""The VFE base config of a preset name or run directory (lenient: retired fields are dropped)."""
    from vfe3.legacy_config import lenient_config_from_serialized

    path = Path(BASE_PRESETS.get(base, base))
    record = json.loads((path / "config.json").read_text(encoding="utf-8"))
    payload = record.get("config", record)
    cfg, dropped = lenient_config_from_serialized(payload, source=str(path / "config.json"))
    if dropped:
        logging.getLogger("train_baseline").info(
            "base %s: dropped retired config fields %s", path.name, sorted(dropped))
    return cfg


def check_base_data(
    base:       str,

    *,
    dataset:    str,
    seeds:      Sequence[int],
    data_seed:  Optional[int],
    max_tokens: Optional[int],
    logger:     Optional[logging.Logger] = None,
) -> None:
    r"""Refuse a launch whose data differs from the base run's (audit 2026-10-12 #5).

    ``dataset`` is compared with the base ``config.json`` ``"dataset"``; its cache tokenizer tag,
    ``max_tokens`` and each seed's data-order seed with the base ``provenance.json`` ``"tokenizer_tag"``,
    ``"max_tokens"`` and ``"data_seed"``. The data-order seed is resolved as :func:`run_baseline` resolves
    it (``data_seed``, or the model seed when None). ``data_seed=None`` also matches a base whose recorded
    data seed equals its model seed, the same model-seed rule applied per seed. A field the base did not
    record is logged as unchecked; any recorded disagreement raises ``ValueError``.
    """
    logger = logger or logging.getLogger("train_baseline")
    path = Path(BASE_PRESETS.get(base, base))
    record = json.loads((path / "config.json").read_text(encoding="utf-8"))
    prov_path = path / "provenance.json"
    prov: Dict[str, Any] = (json.loads(prov_path.read_text(encoding="utf-8")) if prov_path.is_file() else {})
    mismatches = []
    unchecked = []
    if record.get("dataset") is None:
        unchecked.append("dataset")
    elif record["dataset"] != dataset:
        mismatches.append(f"dataset {dataset!r} != base {record['dataset']!r}")
    if prov.get("tokenizer_tag") is None:
        unchecked.append("tokenizer_tag")
    elif prov["tokenizer_tag"] != _tokenizer_tag(dataset):
        mismatches.append(f"tokenizer {_tokenizer_tag(dataset)!r} != base {prov['tokenizer_tag']!r}")
    if "max_tokens" not in prov:
        unchecked.append("max_tokens")
    elif prov["max_tokens"] != max_tokens:
        mismatches.append(f"max_tokens {max_tokens!r} != base {prov['max_tokens']!r}")
    base_data_seed = prov.get("data_seed")
    if base_data_seed is None:
        unchecked.append("data_seed")
    else:
        follows_model_seed = data_seed is None and base_data_seed == prov.get("seed")
        for seed in seeds:
            resolved = int(seed) if data_seed is None else int(data_seed)
            if resolved != base_data_seed and not follows_model_seed:
                mismatches.append(f"seed {seed}: data seed {resolved} != base {base_data_seed}")
    if mismatches:
        raise ValueError(f"base {path}: the baseline data differ from the base run's: " + "; ".join(mismatches))
    if unchecked:
        logger.warning("base %s records no %s; those data settings are unchecked", path.name, unchecked)


def _select_loader(
    dataset:   str,
    cfg:       BaselineConfig,

    *,
    split:     str,
    data_seed: int,
    max_tokens: Optional[int] = None,
) -> DataLoader:
    r"""The loader of ``train_vfe3._select_loader``, with the data seed passed explicitly.

    Train shuffles from a generator seeded with ``data_seed`` and drops its partial last batch;
    validation and test read the whole split in order with the ``eval_stride`` sliding window.
    ``max_tokens`` caps the train split only.
    """
    is_train = split == "train"
    cap = max_tokens if is_train else None
    gen = torch.Generator().manual_seed(int(data_seed)) if is_train else None
    return make_dataloader(dataset, split, cfg.max_seq_len, cfg.batch_size,
                           shuffle=is_train, drop_last=is_train, max_tokens=cap,
                           stride=(None if is_train else cfg.eval_stride),
                           vocab_size=cfg.vocab_size, generator=gen)


def _default_label(cfg: BaselineConfig, dataset: str) -> str:
    _validate_path_component(dataset, "dataset")
    table = "_block_read" if cfg.per_type_table == "block_read" else ""
    return f"{dataset}_K{cfg.embed_dim}_baseline{table}_s{cfg.seed}"


def _reserve_directory(root: Path, base: str) -> Path:
    r"""Atomically reserve ``root/base`` (numeric suffixes on collision)."""
    root.mkdir(parents=True, exist_ok=True)
    suffix = 1
    while True:
        candidate = root / (base if suffix == 1 else f"{base}_{suffix}")
        try:
            candidate.mkdir(exist_ok=False)
        except FileExistsError:
            suffix += 1
            continue
        return candidate


def _rename_by_test_ppl(run_dir: Path, label: str, test_ppl: Optional[float],
                        logger: logging.Logger, *, prefix: str = "") -> Path:
    r"""Rename ``run_dir`` to ``<prefix><ppl:.2f>_<label>`` (unchanged when the PPL is missing or the OS refuses)."""
    if test_ppl is None or not math.isfinite(float(test_ppl)) or not run_dir.exists():
        return run_dir
    target = run_dir.parent / f"{prefix}{float(test_ppl):.2f}_{label}"
    index = 2
    while target.exists():
        target = run_dir.parent / f"{prefix}{float(test_ppl):.2f}_{label}_{index}"
        index += 1
    try:
        run_dir.rename(target)
    except OSError as exc:
        logger.warning("could not rename run dir to %s (%s); kept %s", target.name, exc, run_dir.name)
        return run_dir
    return target


def run_baseline(
    cfg: BaselineConfig,

    *,
    dataset:   str,
    out_root:  "str | Path",
    seed:      int,
    data_seed: Optional[int] = None,
    label:     Optional[str] = None,
    max_tokens: Optional[int] = None,
    device:    "str | torch.device | None" = None,
    logger:    Optional[logging.Logger] = None,
    generate_samples: bool = True,
    score_test: bool = True,
) -> Path:
    r"""Train and finalize one baseline run; return its final run directory.

    With ``score_test=False`` (learning-rate sweep arms) the run is finalized without the test
    split and named ``val<best_val_ppl>_<label>``, so selection never sees a test number
    (audit 2026-10-07 F12/F83).

    ``cfg`` with ``seed`` applied is built into :class:`vfe3.baseline.BaselineTransformer`, trained by
    :func:`vfe3.train.train` on the ``train_vfe3`` loaders (data order from ``data_seed``, the model
    seed when None), validated, and finalized by :func:`vfe3.run_artifacts.finalize_run` (test at the
    final and at the validation-selected weights, training-sample CE, provenance, summary,
    resources). The global RNG is reseeded after the build, as in ``train_vfe3._run_once``.
    """
    from vfe3.run_artifacts import RunArtifacts, finalize_run

    logger = logger or logging.getLogger("train_baseline")
    cfg = replace(cfg, seed=int(seed))
    data_seed = int(seed) if data_seed is None else int(data_seed)
    device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    model = BaselineTransformer(cfg).to(device)
    train_loader = _select_loader(dataset, cfg, split="train", data_seed=data_seed, max_tokens=max_tokens)
    val_loader = _select_loader(dataset, cfg, split="validation", data_seed=data_seed)
    val_tpc = _tokens_per_char(dataset, "validation")
    run_label = label or _default_label(cfg, dataset)
    from datetime import datetime
    run_dir = _reserve_directory(Path(out_root), f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{run_label}")
    artifacts = RunArtifacts(run_dir, cfg, model, dataset=dataset, device=str(device),
                             timestamp=datetime.now().isoformat(timespec="seconds"))
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Baseline %s | %s params (context processing %s) | K=%d H=%d L=%d e=%d table=%s | %s",
                run_label, f"{n_params:,}", f"{model.context_processing_parameter_count():,}",
                cfg.embed_dim, cfg.n_heads, cfg.n_layers, cfg.mlp_expansion, cfg.per_type_table,
                run_dir)
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    started = time.perf_counter()
    final_validation: Dict[str, Optional[float]] = {}
    losses = train(model, train_loader, cfg, n_steps=cfg.max_steps, grad_clip=cfg.grad_clip,
                   log_interval=cfg.log_interval, eval_interval=cfg.eval_interval,
                   val_loader=val_loader, tokens_per_char=val_tpc, device=device, logger=logger,
                   artifacts=artifacts, _final_validation_out=final_validation,
                   generate_samples=generate_samples, sample_dataset=dataset)
    wall = time.perf_counter() - started
    if final_validation:
        iter(val_loader)                     # the omitted pass's CPU base-seed draw, as train_vfe3
        metrics = final_validation
    else:
        metrics = evaluate(model, val_loader, tokens_per_char=val_tpc, device=device)
    logger.info("Final (val) | CE: %.4f | PPL: %.2f", metrics["ce"], metrics["ppl"])
    test_loader = (_select_loader(dataset, cfg, split="test", data_seed=data_seed) if score_test
                   else None)
    results = finalize_run(
        model, artifacts, cfg, final_validation=metrics, tokens_per_char=_tokens_per_char(dataset, "test"),
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, losses=losses,
        data_seed=data_seed, max_tokens=max_tokens, tokenizer_tag=_tokenizer_tag(dataset),
        device=device, wall_time=wall, logger=logger)
    if score_test:
        final_dir = _rename_by_test_ppl(run_dir, run_label, results.get("test_ppl"), logger)
    else:
        final_dir = _rename_by_test_ppl(run_dir, run_label, results.get("best_val_ppl"), logger, prefix="val")
    logger.info("Artifacts written to %s", final_dir)
    return final_dir


def lr_sweep(
    cfg:  BaselineConfig,
    lrs:  Sequence[float],

    *,
    dataset:   str,
    out_root:  "str | Path",
    seed:      int,
    data_seed: Optional[int] = None,
    max_steps: int,
    **run_kwargs: Any,
) -> Dict[str, Any]:
    r"""Short runs at each peak learning rate; the winner has the lowest best validation perplexity.

    Every run keeps the protocol of ``cfg`` except ``lr`` and ``max_steps`` (the warmup is capped at
    ``max_steps``); selection reads only ``summary.json`` ``best_val_ppl`` (validation).
    """
    rows = []
    for lr in lrs:
        swept = replace(cfg, lr=float(lr), max_steps=int(max_steps),
                        warmup_steps=min(cfg.warmup_steps, int(max_steps)),
                        eval_interval=min(cfg.eval_interval, int(max_steps)) or int(max_steps),
                        checkpoint_interval=0)
        run_dir = run_baseline(swept, dataset=dataset, out_root=out_root, seed=seed, data_seed=data_seed,
                               label=f"lr_sweep_{lr:g}_s{seed}", score_test=False, **run_kwargs)
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        rows.append({"lr": float(lr), "best_val_ppl": summary.get("best_val_ppl"), "run_dir": str(run_dir)})
    finite = [row for row in rows if isinstance(row["best_val_ppl"], (int, float))
              and math.isfinite(row["best_val_ppl"])]
    best = min(finite, key=lambda row: row["best_val_ppl"]) if finite else None
    return {"rows": rows, "best_lr": (best["lr"] if best else None), "selection": "best_val_ppl",
            "max_steps": int(max_steps)}


def calibrate_widths(
    cfg:     BaselineConfig,
    widths:  Sequence[int],

    *,
    dataset:      str,
    out_root:     "str | Path",
    seed:         int,
    steps:        int,
    log_interval: int = 100,
    data_seed:    Optional[int] = None,
    max_tokens:   Optional[int] = None,
    device:       "str | torch.device | None" = None,
    logger:       Optional[logging.Logger] = None,
) -> Dict[int, Dict[str, Any]]:
    r"""Training-only throughput runs at each width (no evaluation, no finalize, no model files).

    Each run writes ``metrics.csv`` with the per-window ``train_step_tokens_per_s`` and
    :data:`vfe3.baseline.CALIBRATION_RECORD` with the device it ran on (a calibration run is not
    finalized, so it has no ``provenance.json``); the returned mapping holds
    :func:`vfe3.baseline.calibrated_tokens_per_s` of each run (the median window).
    """
    from vfe3.run_artifacts import RunArtifacts

    logger = logger or logging.getLogger("train_baseline")
    device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    data_seed = int(seed) if data_seed is None else int(data_seed)
    out: Dict[int, Dict[str, Any]] = {}
    for width in widths:
        wide = replace(cfg.with_width(int(width)), seed=int(seed), max_steps=int(steps),
                       warmup_steps=min(cfg.warmup_steps, int(steps)), log_interval=int(log_interval),
                       eval_interval=0, checkpoint_interval=0)
        seed_everything(wide.seed, deterministic=wide.deterministic)
        model = BaselineTransformer(wide).to(device)
        loader = _select_loader(dataset, wide, split="train", data_seed=data_seed, max_tokens=max_tokens)
        run_dir = _reserve_directory(Path(out_root), f"calibration_K{width}_s{seed}")
        artifacts = RunArtifacts(run_dir, wide, model, dataset=dataset, device=str(device),
                                 save_model_artifacts=False)
        seed_everything(wide.seed, deterministic=wide.deterministic)
        train(model, loader, wide, n_steps=wide.max_steps, grad_clip=wide.grad_clip,
              log_interval=wide.log_interval, eval_interval=0, device=device, logger=logger,
              artifacts=artifacts, generate_samples=False)
        device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else device.type
        (run_dir / CALIBRATION_RECORD).write_text(json.dumps(
            {"device_name": device_name, "device": str(device), "torch_version": torch.__version__,
             "width": int(width), "steps": int(steps), "dataset": dataset, "data_seed": data_seed},
            indent=1), encoding="utf-8")
        out[int(width)] = calibrated_tokens_per_s(run_dir)
        logger.info("calibration K=%d: %.0f training tokens/s (%s)", width, out[int(width)]["tokens_per_s"],
                    run_dir)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def compute_matched_config(
    cfg: BaselineConfig,

    *,
    evaluated_run:       "str | Path",
    calibration_runs:    Mapping[int, "str | Path"],
    unique_train_tokens: int,
) -> Dict[str, Any]:
    r"""The compute-matched arm: measured budget, calibrated throughputs, the pre-declared width rule.

    Each calibration run must carry ``cfg.with_width(w)`` up to the calibration protocol fields, and
    must have run on the budget run's device; a mismatch raises (audit 2026-10-07 F84).
    """
    budget = measured_train_device_hours(evaluated_run)
    calibrated: Dict[int, Dict[str, Any]] = {}
    for w, path in calibration_runs.items():
        mismatched = calibration_mismatch(cfg, path, int(w))
        if mismatched:
            raise ValueError(f"calibration run {path} (K={w}) differs from the matched arm in {mismatched}")
        calibrated[int(w)] = calibrated_tokens_per_s(path)
        if (budget.get("device_name") is None
                or calibrated[int(w)].get("device_name") != budget.get("device_name")):
            raise ValueError(f"calibration run {path} ran on {calibrated[int(w)].get('device_name')!r}, the "
                             f"evaluated run on {budget.get('device_name')!r}; throughput must share a device")
    throughput = {w: row["tokens_per_s"] for w, row in calibrated.items()}
    rule = width_ladder_rule(cfg, hours=budget["train_hours"], tokens_per_s=throughput,
                             unique_train_tokens=int(unique_train_tokens))
    rule["budget"] = budget
    return rule


def main() -> None:
    r"""Click-to-run: optional lr sweep, calibration and compute matching, then one run per seed."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("train_baseline")
    base = load_base_config(BASE)
    check_base_data(BASE, dataset=DATASET, seeds=SEEDS, data_seed=DATA_SEED, max_tokens=MAX_TOKENS, logger=logger)
    fallback = {} if getattr(base, "rope_mode", "none") != "none" else dict(POSITION_FALLBACK)
    cfg = BaselineConfig.matched_to(base, per_type_table=PER_TYPE_TABLE, **{**fallback, **OVERRIDES})
    logger.info("baseline position: prior=%s, pos_rotation=%s (rope_base %g)",
                cfg.beta_attention_prior, cfg.pos_rotation, cfg.rope_base)
    out_root = Path(OUT_ROOT)
    run_kwargs = dict(max_tokens=MAX_TOKENS, device=DEVICE, logger=logger)
    if CALIBRATE_WIDTHS:
        calibration = calibrate_widths(cfg, CALIBRATE_WIDTHS, dataset=DATASET, out_root=out_root / "calibration",
                                       seed=SEEDS[0], steps=CALIBRATION_STEPS,
                                       log_interval=CALIBRATION_LOG_INTERVAL, data_seed=DATA_SEED, **run_kwargs)
        logger.info("calibration: %s", {w: round(v["tokens_per_s"]) for w, v in calibration.items()})
    if COMPUTE_MATCH is not None:
        unique = COMPUTE_MATCH.get("unique_train_tokens") or cached_token_count(DATASET, "train")
        rule = compute_matched_config(cfg, evaluated_run=COMPUTE_MATCH["evaluated_run"],
                                      calibration_runs=COMPUTE_MATCH["calibration_runs"],
                                      unique_train_tokens=int(unique))
        cfg = rule.pop("config")
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "compute_match_rule.json").write_text(
            json.dumps({**rule, "config": cfg.serialize()}, indent=2, default=str), encoding="utf-8")
        logger.info("compute-matched width K=%d, max_steps=%d (%.2f passes; cap binds: %s)",
                    cfg.embed_dim, cfg.max_steps,
                    next(r["passes"] for r in rule["candidates"] if r["width"] == cfg.embed_dim),
                    rule["cap_binds"])
    if LR_SWEEP:
        sweep = lr_sweep(cfg, LR_SWEEP, dataset=DATASET, out_root=out_root / "lr_sweep", seed=SEEDS[0],
                         data_seed=DATA_SEED, max_steps=LR_SWEEP_STEPS, **run_kwargs)
        (out_root / "lr_sweep.json").parent.mkdir(parents=True, exist_ok=True)
        (out_root / "lr_sweep.json").write_text(json.dumps(sweep, indent=2), encoding="utf-8")
        if sweep["best_lr"] is not None:
            cfg = replace(cfg, lr=sweep["best_lr"])
            logger.info("lr sweep: selected lr=%g on validation", sweep["best_lr"])
    for seed in SEEDS:
        run_baseline(cfg, dataset=DATASET, out_root=out_root, seed=int(seed), data_seed=DATA_SEED,
                     label=(f"{LABEL}_s{seed}" if LABEL else None), **run_kwargs)


if __name__ == "__main__":
    main()
