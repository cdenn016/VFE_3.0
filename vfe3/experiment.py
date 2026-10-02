r"""One complete VFE training run as a library call: build, train, validate, test, finalize.

:func:`run_vfe` is the library equivalent of ``train_vfe3._run_once``. It takes every input as an
argument (the config, the dataset, the output root, the model seed, the data-order seed and the
run label), so experiment launchers such as ``paper_arms.py`` can run many configurations in one
process without editing or importing the click-to-run launcher and its module globals.

The sequence is the launcher's, step for step. The global RNG is seeded from ``seed`` before the
model is built and again after it (so later global draws do not depend on how much RNG the model's
construction consumed); the training loader shuffles with its own generator seeded from
``data_seed``, so the batch order is a function of the data seed alone; validation and test read the
whole split in order with the config's evaluation stride. After training, the final validation
handed back by the loop (or a fresh evaluation) is logged, :func:`vfe3.run_artifacts.finalize_run`
scores the test split and writes the run's records, and the directory
``<out_root>/<timestamp>_<label>`` is renamed to ``<out_root>/<test PPL>_<label>``.
"""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from torch.utils.data import DataLoader

from vfe3.config import VFE3Config, config_from_serialized, serialize_config

__all__ = ["run_vfe", "reserve_directory", "rename_run_by_test_ppl"]


def reserve_directory(root: Path, base: str) -> Path:
    r"""Atomically create ``root/base`` (``base_2``, ``base_3``, ... on a collision) and return it."""
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


def rename_run_by_test_ppl(
    run_dir:  Path,
    label:    str,
    test_ppl: Optional[float],
    logger:   logging.Logger,
) -> Path:
    r"""Rename ``run_dir`` to ``<parent>/<test_ppl:.2f>_<label>`` (``_2``, ... on a clash).

    Model selection is validation-only; the name is a sorting convenience applied after the test
    evaluation. The directory keeps its timestamped name when the perplexity is missing or not
    finite, or when the operating system refuses the move (the results are already on disk).
    """
    if test_ppl is None or not math.isfinite(test_ppl) or not run_dir.exists():
        return run_dir
    target = run_dir.parent / f"{test_ppl:.2f}_{label}"
    index = 2
    while target.exists():
        target = run_dir.parent / f"{test_ppl:.2f}_{label}_{index}"
        index += 1
    try:
        run_dir.rename(target)
    except OSError as exc:
        logger.warning("could not rename run dir to %s (%s); kept %s", target.name, exc, run_dir.name)
        return run_dir
    logger.info("Renamed run dir -> %s", target.name)
    return target


def _with_seed(cfg: VFE3Config, seed: int) -> VFE3Config:
    r"""``cfg`` with ``seed`` replaced, rebuilt through the serialized form."""
    if cfg.seed == seed:
        return cfg
    return config_from_serialized({**serialize_config(cfg), "seed": seed}, source="run_vfe seed override")


def _loader(
    dataset:   str,
    cfg:       VFE3Config,
    split:     str,
    data_seed: int,
) -> DataLoader:
    r"""The loader ``train_vfe3._select_loader`` builds for ``split`` (uncapped streams).

    Train shuffles with a generator seeded from ``data_seed`` and drops its partial last batch;
    validation and test read the whole split in order, windowed at ``cfg.eval_stride``.
    """
    from vfe3.data.datasets import make_dataloader

    is_train = split == "train"
    generator = torch.Generator().manual_seed(int(data_seed)) if is_train else None
    return make_dataloader(dataset, split, cfg.max_seq_len, cfg.batch_size,
                           shuffle=is_train, drop_last=is_train, max_tokens=None,
                           stride=(None if is_train else cfg.eval_stride),
                           vocab_size=cfg.vocab_size, generator=generator)


def run_vfe(
    cfg:       VFE3Config,

    *,
    dataset:   str,
    out_root:  Union[str, Path],
    seed:      int,
    data_seed: int,
    label:     str,
    device:    Optional[Union[str, torch.device]] = None,
    logger:    Optional[logging.Logger] = None,
) -> Path:
    r"""Train ``cfg`` on ``dataset`` and finalize the run; return its (renamed) run directory.

    ``seed`` replaces ``cfg.seed`` (the model-initialization and global RNG seed); ``data_seed``
    seeds the training shuffle generator only. ``label`` names the run directory
    ``<out_root>/<timestamp>_<label>`` while training and ``<out_root>/<test PPL:.2f>_<label>``
    after finalization. ``device`` defaults to CUDA when available.
    """
    from vfe3.data.datasets import _tokenizer_tag, _validate_path_component, tokens_per_char
    from vfe3.model.model import VFEModel
    from vfe3.run_artifacts import RunArtifacts, finalize_run
    from vfe3.runtime import seed_everything
    from vfe3.train import evaluate, train

    for name, value in (("seed", seed), ("data_seed", data_seed)):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be an exact non-negative integer, got {value!r}")
    _validate_path_component(dataset, "dataset")
    _validate_path_component(label, "label")
    logger = logger or logging.getLogger("vfe3.experiment")
    device = torch.device(device if device is not None else ("cuda" if torch.cuda.device_count() > 0 else "cpu"))
    cfg = _with_seed(cfg, seed)

    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    model = VFEModel(cfg).to(device)
    train_loader = _loader(dataset, cfg, "train", data_seed)
    val_loader = _loader(dataset, cfg, "validation", data_seed)
    val_tpc = tokens_per_char(dataset, "validation")

    run_dir = reserve_directory(Path(out_root), f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{label}")
    artifacts = RunArtifacts(run_dir, cfg, model, dataset=dataset, device=device,
                             timestamp=datetime.now().isoformat(timespec="seconds"))
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("run %s | %s | %s params | K=%d L=%d heads=%d | steps=%d batch=%d N=%d | seed=%d "
                "data_seed=%d | %s", run_dir, dataset, f"{n_params:,}", cfg.embed_dim, cfg.n_layers,
                cfg.n_heads, cfg.max_steps, cfg.batch_size, cfg.max_seq_len, cfg.seed, data_seed, device)

    # Reseed after construction, exactly as the launcher does, so the loop's global draws do not
    # depend on the config-dependent amount of RNG VFEModel(cfg) consumed.
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    started = time.perf_counter()
    final_validation: Dict[str, Optional[float]] = {}
    losses = train(
        model, train_loader, cfg,
        n_steps=cfg.max_steps,
        grad_clip=cfg.grad_clip,
        log_interval=cfg.log_interval,
        eval_interval=cfg.eval_interval,
        val_loader=val_loader,
        tokens_per_char=val_tpc,
        device=device,
        logger=logger,
        artifacts=artifacts,
        _final_validation_out=final_validation,
        sample_dataset=dataset,
    )
    wall = time.perf_counter() - started

    if final_validation:
        # The launcher's handoff: advance the validation loader's base-seed draw that the omitted
        # evaluation pass would have made (a zero-worker sequential loader draws no RNG otherwise).
        iter(val_loader)
        metrics = final_validation
    else:
        metrics = evaluate(model, val_loader, tokens_per_char=val_tpc, device=device)
    logger.info("Final (val) | CE: %.4f | PPL: %.1f | BPT: %.4f | BPC: %s",
                metrics["ce"], metrics["ppl"], metrics["bits_per_token"],
                f"{float(metrics['bpc']):.4f}" if metrics["bpc"] is not None else "unavailable")

    test_loader = _loader(dataset, cfg, "test", data_seed)
    results = finalize_run(
        model, artifacts, cfg,
        final_validation=metrics,
        tokens_per_char=tokens_per_char(dataset, "test"),
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        losses=losses,
        data_seed=int(data_seed),
        max_tokens=None,
        tokenizer_tag=_tokenizer_tag(dataset),
        device=device,
        wall_time=wall,
        logger=logger,
    )
    run_dir = rename_run_by_test_ppl(run_dir, label, results.get("test_ppl"), logger)
    logger.info("Artifacts written to %s", run_dir)
    return run_dir

