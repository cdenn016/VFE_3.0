r"""One complete VFE training run as a library call: build, train, validate, test, finalize.

:func:`run_vfe` is the library equivalent of ``train_vfe3._run_once``. It takes every input as an
argument (the config, the dataset, the output root, the model seed, the data-order seed and the
run label), so experiment launchers such as ``paper_arms.py`` can run many configurations in one
process without editing or importing the click-to-run launcher and its module globals.

The sequence is the launcher's, and the trained weights and test scores match it bitwise. The global
RNG is seeded from ``seed`` before the model is built and again after it (so later global draws do
not depend on how much RNG the model's construction consumed); the training loader shuffles with its
own generator seeded from ``data_seed``, so the batch order is a function of the data seed alone;
validation and test read the whole split in order with the config's evaluation stride. All three
splits are opened before training (building a sequential loader draws no RNG), so a missing test
cache fails at launch rather than after the training budget (audit 2026-10-16 F122). After training,
the final validation handed back by the loop (or a fresh evaluation) is logged, :func:`vfe3.run_artifacts.finalize_run`
scores the test split and writes the run's records, and the directory
``<out_root>/<timestamp>_<label>`` is renamed to ``<out_root>/<test PPL>_<label>``.

:func:`saved_train_window_mode` recovers the training-window origin a saved run trained with. Every run
this code trains draws a per-epoch random origin (:data:`CURRENT_TRAIN_WINDOW_MODE`); the config loaders
drop a saved ``train_window_offset`` silently, so launchers that reuse or pool saved runs read the
protocol here (audit 2026-10-16 F59).

When training or finalization raises, :func:`run_vfe` (and ``train_baseline.run_baseline``) writes
:data:`FAILURE_RECORD` into the run directory with :func:`write_failure_record` before re-raising, so a
crashed run is a recorded failure rather than an unfinished one, and a relaunch does not re-roll its seed
pair (audit 2026-10-17 item 28). :func:`read_failure_record` reads it back.
"""

from __future__ import annotations

import csv
import functools
import json
import logging
import math
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader

from vfe3.config import VFE3Config, config_from_serialized, serialize_config
from vfe3.runtime import restored_fp32_matmul_precision

__all__ = ["run_vfe", "reserve_directory", "rename_run_by_test_ppl", "saved_train_window_mode",
           "write_failure_record", "read_failure_record", "FAILURE_RECORD",
           "CURRENT_TRAIN_WINDOW_MODE", "TRAIN_WINDOW_OFFSET_RETIREMENT_COMMIT"]

# The crash record a library run writes before re-raising (audit 2026-10-17 item 28).
FAILURE_RECORD = "failure.json"

# The training-window origin. ``train_window_offset`` existed from 2026-10-09 (default ``'fixed'``; before
# then every training window started at a fixed origin) until the commit below hardcoded the per-epoch
# random origin on 2026-10-12. Every run trained by this code draws that origin.
TRAIN_WINDOW_OFFSET_RETIREMENT_COMMIT = "1b50c6ea4b16223b68ded489ffb131e9efdd8488"
CURRENT_TRAIN_WINDOW_MODE = "per_epoch_random"
_TRAIN_WINDOW_MODES = ("fixed", "per_epoch_random")
_REPO_ROOT = Path(__file__).resolve().parent.parent


@functools.lru_cache(maxsize=None)
def _descends_from_window_retirement(git_sha: str) -> Optional[bool]:
    r"""Whether commit ``git_sha`` contains :data:`TRAIN_WINDOW_OFFSET_RETIREMENT_COMMIT`, or None when git
    cannot tell (no git, an unknown commit, a repository without the retirement commit)."""
    git = shutil.which("git")
    if git is None:
        return None
    from vfe3.run_artifacts import _git_environment

    try:
        completed = subprocess.run(
            [git, "-c", f"safe.directory={_REPO_ROOT.as_posix()}", "merge-base", "--is-ancestor",
             TRAIN_WINDOW_OFFSET_RETIREMENT_COMMIT, git_sha],
            cwd=str(_REPO_ROOT), env=_git_environment(git), stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return {0: True, 1: False}.get(completed.returncode)


def saved_train_window_mode(
    config_raw: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> Tuple[str, str]:
    r"""The training-window origin a saved run trained with, and the evidence: ``(mode, evidence)``.

    ``mode`` is ``"fixed"`` (window :math:`i` starts at :math:`i s` in every epoch), ``"per_epoch_random"``
    (at :math:`o + i s` with an origin :math:`o` drawn per epoch) or ``"unknown"``. The two protocols present
    different training windows, so their runs are not replicates of one configuration. The saved
    ``train_window_offset`` (``config_raw``, the run's serialized config) decides when present, then a
    ``provenance["train_window_mode"]`` record. Otherwise the run trained before the option existed or after
    its retirement, and the run's commit (``provenance["git_sha"]``) decides: a commit containing
    :data:`TRAIN_WINDOW_OFFSET_RETIREMENT_COMMIT` trained with per-epoch random origins, any other with
    fixed windows.
    """
    if "train_window_offset" in config_raw:
        value = config_raw["train_window_offset"]
        if value in _TRAIN_WINDOW_MODES:
            return str(value), "config.json records train_window_offset"
        return "unknown", f"config.json records train_window_offset={value!r}"
    recorded = provenance.get("train_window_mode")
    if recorded in _TRAIN_WINDOW_MODES:
        return str(recorded), "provenance.json records train_window_mode"
    git_sha = provenance.get("git_sha")
    if not isinstance(git_sha, str) or re.fullmatch(r"[0-9a-fA-F]{7,64}", git_sha) is None:
        return "unknown", "config.json records no train_window_offset and provenance.json no git_sha"
    short = git_sha[:10]
    descends = _descends_from_window_retirement(git_sha.lower())
    if descends is None:
        return "unknown", (f"git cannot place commit {short} relative to the 2026-10-12 retirement of "
                           f"train_window_offset ({TRAIN_WINDOW_OFFSET_RETIREMENT_COMMIT[:10]})")
    if descends:
        return "per_epoch_random", (f"trained at commit {short}, which contains the 2026-10-12 retirement of "
                                    "train_window_offset")
    return "fixed", (f"trained at commit {short}, before the 2026-10-12 retirement of train_window_offset, "
                     "and config.json records no train_window_offset (fixed windows were the only mode "
                     "before 2026-10-09 and the option's default after)")


def _last_logged_step(run_dir: Path) -> Optional[int]:
    r"""The largest finite ``step`` in ``run_dir/metrics.csv``, or None when there is none."""
    try:
        with open(run_dir / "metrics.csv", newline="", encoding="utf-8") as handle:
            steps = []
            for row in csv.DictReader(handle):
                try:
                    value = float(row.get("step") or "nan")
                except ValueError:
                    continue
                if math.isfinite(value):
                    steps.append(int(value))
    except (OSError, csv.Error):
        return None
    return max(steps) if steps else None


def write_failure_record(
    run_dir: Path,
    exc:     BaseException,

    *,
    stage:   str,
    step:    Optional[int] = None,
) -> Optional[Path]:
    r"""Write :data:`FAILURE_RECORD` into ``run_dir``: the exception type and message, the stage and step.

    ``step`` defaults to ``exc.step`` when the exception carries one, else to the last step logged in
    ``metrics.csv`` (``step_source`` says which; None when neither is known). Never raises: a record
    that cannot be written is skipped, so the original exception still propagates.
    """
    step_source = "argument" if step is not None else None
    if step is None and isinstance(getattr(exc, "step", None), int):
        step, step_source = int(getattr(exc, "step")), "exception"
    if step is None:
        step = _last_logged_step(run_dir)
        step_source = "last_logged_step" if step is not None else None
    record = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc), "stage": stage,
              "step": step, "step_source": step_source,
              "timestamp": datetime.now().isoformat(timespec="seconds")}
    try:
        from vfe3.run_artifacts import _write_json_atomic

        return _write_json_atomic(Path(run_dir) / FAILURE_RECORD, record)
    except Exception:                                   # the crash itself must still propagate
        return None


def read_failure_record(run_dir: Path) -> Optional[Dict[str, Any]]:
    r"""The :data:`FAILURE_RECORD` of ``run_dir`` as a mapping, or None when absent or unreadable."""
    try:
        record = json.loads((Path(run_dir) / FAILURE_RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


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

    *,
    prefix:   str = "",
) -> Path:
    r"""Rename ``run_dir`` to ``<parent>/<prefix><test_ppl:.2f>_<label>`` (``_2``, ... on a clash).

    Nothing within the run is selected on test; the name is a sorting convenience applied after
    the test evaluation. Across runs the name exposes the test perplexity, so recipes chosen by
    comparing run names were chosen on the test split. The directory keeps its timestamped name when the perplexity is missing or not
    finite, or when the operating system refuses the move (the results are already on disk).
    ``prefix`` marks a name that carries another perplexity: ``train_baseline`` names a learning-rate
    sweep arm, which never scores test, ``val<best_val_ppl:.2f>_<label>``.
    """
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
    logger.info("Renamed run dir -> %s", target.name)
    return target


def _with_seed(cfg: VFE3Config, seed: int) -> VFE3Config:
    r"""``cfg`` with ``seed`` replaced, rebuilt through the serialized form."""
    if cfg.seed == seed:
        return cfg
    return config_from_serialized({**serialize_config(cfg), "seed": seed}, source="run_vfe seed override")


def _without_resume(cfg: VFE3Config, label: str, logger: logging.Logger) -> VFE3Config:
    r"""``cfg`` with ``resume_from`` and ``trust_resume_checkpoint`` cleared: :func:`run_vfe` trains from step 0.

    A config resolved from a resumed base record still carries that record's ``resume_from``, which
    ``train()`` would otherwise follow into the base's checkpoint (audit 2026-10-08 P2-9).
    """
    if cfg.resume_from is None and not cfg.trust_resume_checkpoint:
        return cfg
    logger.warning("run %s: clearing resume_from=%r inherited from the base config; run_vfe trains "
                   "from step 0", label, cfg.resume_from)
    return config_from_serialized({**serialize_config(cfg), "resume_from": None,
                                   "trust_resume_checkpoint": False}, source="run_vfe resume reset")


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


@restored_fp32_matmul_precision()         # a None arm must not inherit the previous arm's pinned matmul policy
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

    An exception during training or finalization writes :data:`FAILURE_RECORD` into the run directory
    (:func:`write_failure_record`) and is re-raised.

    ``seed`` replaces ``cfg.seed`` (the model-initialization and global RNG seed); ``data_seed``
    seeds the training shuffle generator only. ``label`` names the run directory
    ``<out_root>/<timestamp>_<label>`` while training and ``<out_root>/<test PPL:.2f>_<label>``
    after finalization. ``device`` defaults to CUDA when available.
    """
    from vfe3.data.datasets import _validate_path_component, tokens_per_char
    from vfe3.model.model import VFEModel
    from vfe3.run_artifacts import RunArtifacts
    from vfe3.runtime import seed_everything

    for name, value in (("seed", seed), ("data_seed", data_seed)):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be an exact non-negative integer, got {value!r}")
    _validate_path_component(dataset, "dataset")
    _validate_path_component(label, "label")
    logger = logger or logging.getLogger("vfe3.experiment")
    device = torch.device(device if device is not None else ("cuda" if torch.cuda.device_count() > 0 else "cpu"))
    cfg = _with_seed(cfg, seed)
    cfg = _without_resume(cfg, label, logger)

    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    model = VFEModel(cfg).to(device)
    train_loader = _loader(dataset, cfg, "train", data_seed)
    val_loader = _loader(dataset, cfg, "validation", data_seed)
    val_tpc = tokens_per_char(dataset, "validation")
    # The test split is opened here, with the others, so a missing or unreadable test cache fails at
    # launch rather than after the whole training budget (audit 2026-10-16 F122). Building a
    # sequential loader draws no RNG, so the run is unchanged.
    test_loader = _loader(dataset, cfg, "test", data_seed)
    test_tpc = tokens_per_char(dataset, "test")

    run_dir = reserve_directory(Path(out_root), f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{label}")
    artifacts = RunArtifacts(run_dir, cfg, model, dataset=dataset, device=device,
                             timestamp=datetime.now().isoformat(timespec="seconds"))
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("run %s | %s | %s params | K=%d L=%d heads=%d | steps=%d batch=%d N=%d | seed=%d "
                "data_seed=%d | %s", run_dir, dataset, f"{n_params:,}", cfg.embed_dim, cfg.n_layers,
                cfg.n_heads, cfg.max_steps, cfg.batch_size, cfg.max_seq_len, cfg.seed, data_seed, device)

    try:
        return _train_and_finalize(model, cfg, artifacts, run_dir, label, dataset=dataset, data_seed=data_seed,
                                   train_loader=train_loader, val_loader=val_loader, val_tpc=val_tpc,
                                   test_loader=test_loader, test_tpc=test_tpc, device=device, logger=logger)
    except Exception as exc:
        # A crash is a recorded failure of the seed pair, not an unfinished run (audit 2026-10-17 item 28).
        write_failure_record(run_dir, exc, stage="train_or_finalize")
        raise


def _train_and_finalize(
    model:        torch.nn.Module,
    cfg:          VFE3Config,
    artifacts:    Any,
    run_dir:      Path,
    label:        str,

    *,
    dataset:      str,
    data_seed:    int,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    val_tpc:      Any,
    test_loader:  DataLoader,
    test_tpc:     Any,
    device:       torch.device,
    logger:       logging.Logger,
) -> Path:
    r"""The training, final validation, test scoring and rename of :func:`run_vfe`."""
    from vfe3.data.datasets import _tokenizer_tag
    from vfe3.run_artifacts import finalize_run
    from vfe3.runtime import seed_everything
    from vfe3.train import evaluate, train

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

    results = finalize_run(
        model, artifacts, cfg,
        final_validation=metrics,
        tokens_per_char=test_tpc,
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

