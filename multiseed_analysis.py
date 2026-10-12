r"""Across-seed digest for an identical-config multi-seed run.

``train_vfe3.py`` with ``NUM_RUNS``/``SEEDS`` writes one seed-labeled run dir per seed, each holding a
``summary.json`` / ``test_results.json`` (headline scalars), ``provenance.json`` (the ``seed``),
``research.json`` (ECE, frequency-stratified CE, sigma diagnostics), ``metrics.csv`` (the per-step
training curves) and ``metrics_per_layer.csv`` (final per-layer diagnostics). This module aggregates
EVERY one of those across seeds and emits the full figure + data set in one pass:

- scalar headline + research metrics -> across-seed mean / SD / CV table, ``scalar_cv_summary`` figure,
  and a per-metric noise band (the floor every ablation 'win' must clear);
- per-step curves -> across-seed mean +/-1 SD ribbons (one per curated metric + an overview grid);
- per-layer diagnostics -> across-seed per-layer bars with SD error bars;
- machine-readable ``multiseed_summary.json`` / ``.csv`` and a human-readable ``MULTISEED_ANALYSIS.md``.

Click-to-run (project policy: no argparse): edit ``CONFIG`` below, then ``python multiseed_analysis.py``.
``run_root`` resolves a bare run-folder name under ``vfe3_runs/`` (so ``"K=20_GL(10)"`` just works).

Caveat carried into every output: the per-run reseed (train_vfe3) currently shares the data-shuffle
order across seeds, so this SD is the init+optimization spread only -- a LOWER BOUND on deployment
variance. A fixed data-order generator (model-init RNG varies, batch order held) is the companion fix;
see docs/experiments/2026-06-21-experiment-readiness.md (S6).
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import re
import typing
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


from vfe3.process_utils import run_isolated_python


_CHILD_TIMEOUT_SECONDS = 3600      # bound the isolated worker; see run_process_tree
_CHILD_SENTINEL = "_VFE3_MULTISEED_ANALYSIS_CHILD"


def _run_isolated_child() -> int:
    """Run this driver without loading scientific libraries in the parent."""
    return run_isolated_python(__file__, _CHILD_SENTINEL, _CHILD_TIMEOUT_SECONDS, "multi-seed analysis")


if __name__ == "__main__" and os.environ.get(_CHILD_SENTINEL) != "1":
    raise SystemExit(_run_isolated_child())

if os.environ.get(_CHILD_SENTINEL) == "1":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np

from vfe3.paper.runs import load_config
from vfe3.run_artifacts import _write_json_atomic

logger = logging.getLogger(__name__)


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _is_exact_nonnegative_seed(value: Any) -> bool:
    """Whether ``value`` is a plain Python ``int`` seed in the supported domain."""
    return type(value) is int and value >= 0


def _strip_seed_keys(value: Any) -> Any:
    """Recursively remove exact ``seed`` keys while preserving every other serialized value."""
    if isinstance(value, dict):
        return {key: _strip_seed_keys(item) for key, item in value.items() if key != "seed"}
    if isinstance(value, list):
        return [_strip_seed_keys(item) for item in value]
    return value


def _semantic_config(config_path: Path) -> Dict[str, Any]:
    """Migrate one nested or flat artifact (raw values when it cannot be built) and remove only its run seed."""
    try:
        serialized = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"cannot aggregate {config_path.parent}: expected a readable config.json at {config_path}"
        ) from exc
    if not isinstance(serialized, dict):
        raise ValueError(f"cannot aggregate {config_path.parent}: config.json must contain an object")
    raw_config = serialized.get("config", serialized)
    if not isinstance(raw_config, dict):
        raise ValueError(
            f"cannot aggregate {config_path.parent}: config.json['config'] must contain an object"
        )
    # No forward is replayed here, so a run that saved a value this build refuses is read from its raw
    # saved values, as the manuscript assemblers read it (vfe3.paper.runs.load_config).
    cfg, load_mode, dropped = load_config(raw_config, source=str(config_path))
    if cfg is None:
        logger.warning("%s: config could not be built (%s); raw saved values used", config_path, load_mode)
        return _strip_seed_keys(dict(raw_config))
    retired = {f"retired:{key}": value for key, value in dropped.items()}
    return _strip_seed_keys({**_float_fields_as_float(cfg), **retired})


def _float_fields_as_float(config: Any) -> Dict[str, Any]:
    """``asdict(config)`` with exact-``int`` values of float-annotated fields coerced to ``float``.

    ``VFE3Config`` accepts any ``Real`` for its float fields without coercion, so ``rope_fraction=1``
    and ``rope_fraction=1.0`` serialize as ``1`` and ``1.0`` and would otherwise give one semantic
    config two fingerprints. Only fields annotated ``float`` or ``Optional[float]`` are touched;
    ``bool`` values and fields annotated as ``int`` keep their type.
    """
    payload = asdict(config)
    hints = typing.get_type_hints(type(config))
    for name, value in payload.items():
        if type(value) is not int:
            continue
        hint = hints.get(name)
        if hint is float or (
                typing.get_origin(hint) is typing.Union
                and float in typing.get_args(hint)
                and int not in typing.get_args(hint)):
            payload[name] = float(value)
    return payload


def _config_requires_per_layer(config_path: Path) -> bool:
    """Whether this artifact contract requires ``metrics_per_layer.csv``.

    Standard numeric diagnostics are emitted independent of the embedding switch.
    """
    try:
        serialized = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read figure policy from {config_path}") from exc
    if not isinstance(serialized, dict):
        raise ValueError(f"cannot read figure policy from {config_path}: expected an object")
    raw_config = serialized.get("config", serialized)
    if not isinstance(raw_config, dict):
        raise ValueError(
            f"cannot read figure policy from {config_path}: config must contain an object"
        )
    # Standard numeric diagnostics are emitted independent of the embedding switch, so every
    # config this aggregator can read requires the channel. Reading it here also keeps an
    # unreadable config from being silently treated as a complete contract.
    _semantic_config(config_path)
    return True


def _config_fingerprint(config_path: Path) -> str:
    """Deterministic SHA-256 identity of one seed-normalized semantic config."""
    normalized = json.dumps(
        _semantic_config(config_path),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def _assert_homogeneous_configs(run_dirs: List[Path]) -> Optional[str]:
    """Return the shared semantic fingerprint or abort with each distinct fingerprint and path."""
    groups: Dict[str, List[Path]] = {}
    for run_dir in run_dirs:
        config_path = run_dir / "config.json"
        fingerprint = _config_fingerprint(config_path)
        groups.setdefault(fingerprint, []).append(config_path)
    if len(groups) > 1:
        detail = ["mixed semantic config fingerprints; refusing across-seed aggregation:"]
        for fingerprint, paths in sorted(groups.items()):
            detail.extend(f"  {fingerprint}  {path}" for path in paths)
        raise ValueError("\n".join(detail))
    return next(iter(groups), None)


def _as_finite_float(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def aggregate_seed_metric(
    run_root:  Any,
    key:       str = "test_ppl",

    *,
    filename:  str = "summary.json",
    seed_file: str = "config.json",
) -> Dict[str, Any]:
    r"""Scan ``run_root`` for per-seed ``filename``, gather ``key``, return the across-seed summary.

    Returns ``{n, mean, sd, noise_floor, cv, values, seeds}`` with the UNBIASED (ddof=1) SD and the
    :func:`single_seed_noise_floor`; ``sd``/``noise_floor``/``cv`` are NaN for n<2. ``seeds`` reads :func:`_seed_for` (provenance.json -> config.json -> dir name).
    Non-finite or unreadable points are skipped (never crash the aggregation).
    """
    root = Path(run_root)
    return _aggregate_requested_metric(
        root,
        key,
        sources=(filename,),
        seed_file=seed_file,
    )


def single_seed_noise_floor(sd: float, n_sd_seeds: int) -> float:
    r"""Half-width of the 95% band for the difference of two single-seed runs.

    .. math::
        t_{n-1, 0.975} \sqrt{2} s,

    with :math:`s` the across-seed SD estimated from :math:`n` seeds. NaN for :math:`n < 2` or a
    non-finite :math:`s`. A single-seed ablation difference below it is within seed noise.
    """
    if n_sd_seeds < 2 or not math.isfinite(sd):
        return math.nan
    return _student_t_quantile_975(n_sd_seeds - 1) * math.sqrt(2.0) * sd


def _student_t_quantile_975(nu: int) -> float:
    r"""The 0.975 quantile of Student's t with an integer :math:`\nu \geq 1` degrees of freedom, without scipy.

    With :math:`\theta = \arctan(t / \sqrt{\nu})` and :math:`c = \cos\theta`, the two-sided probability
    :math:`A(t) = P(|T| \leq t)` has the finite closed form (Abramowitz and Stegun 1964, 26.7.3-26.7.4)

    .. math::
        A = \frac{2}{\pi}\Big[\theta + \sin\theta \sum_{k=0}^{(\nu-3)/2} a_k c^{2k+1}\Big],
        \quad a_0 = 1,\ a_k = a_{k-1} \frac{2k}{2k+1} \quad (\nu \text{ odd}),

        A = \sin\theta \sum_{k=0}^{(\nu-2)/2} b_k c^{2k},
        \quad b_0 = 1,\ b_k = b_{k-1} \frac{2k-1}{2k} \quad (\nu \text{ even}).

    :math:`A` increases strictly in :math:`\theta \in (0, \pi/2)`, so bisection on :math:`A = 0.95` to
    rounding gives the quantile :math:`\sqrt{\nu} \tan\theta`.
    """
    def coverage(theta: float) -> float:
        s, c = math.sin(theta), math.cos(theta)
        if nu % 2:
            total, term = 0.0, c
            for k in range((nu - 1) // 2):
                total += term
                term *= c * c * (2 * k + 2) / (2 * k + 3)
            return 2.0 / math.pi * (theta + s * total)
        total, term = 0.0, 1.0
        for k in range(nu // 2):
            total += term
            term *= c * c * (2 * k + 1) / (2 * k + 2)
        return s * total

    low, high = 0.0, math.pi / 2
    for _ in range(200):
        mid = 0.5 * (low + high)
        if mid in (low, high):
            break
        low, high = (mid, high) if coverage(mid) < 0.95 else (low, mid)
    return math.sqrt(nu) * math.tan(0.5 * (low + high))


def _summarize(values: List[float]) -> Dict[str, Any]:
    r"""Across-seed ``{n, mean, sd, noise_floor, cv, values}`` with the UNBIASED (ddof=1) SD and the
    :func:`single_seed_noise_floor` (both NaN for n<2)."""
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": math.nan, "sd": math.nan, "noise_floor": math.nan,
                "cv": math.nan, "values": []}
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) if n >= 2 else math.nan
    cv = (sd / abs(mean)) if (n >= 2 and mean != 0.0) else math.nan
    return {"n": n, "mean": mean, "sd": sd, "noise_floor": single_seed_noise_floor(sd, n),
            "cv": cv, "values": values}


def _resolve_run_root(
    run_root: Any,

    *,
    search_dirs: tuple = ("vfe3_runs",),
) -> Path:
    r"""Resolve ``run_root`` to a directory: the literal if it exists, else ``<search_dir>/run_root``
    for the first ``search_dir`` that contains it, else the literal (the caller reports "nothing
    found"). Lets the bare run name ``"K=20_GL(10)"`` resolve to ``vfe3_runs/K=20_GL(10)``.
    """
    p = Path(run_root)
    if p.exists():
        return p
    for s in search_dirs:
        cand = Path(s) / run_root
        if cand.exists():
            return cand
    return p


def _seed_for(
    run_dir: Any,

    *,
    config_name: str = "config.json",
    prov_name:   str = "provenance.json",
) -> Optional[int]:
    r"""Return the one seed on which every present run identity agrees.

    Provenance, flat or nested config, and the canonical ``_s<NN>`` directory suffix are independent
    join keys. Missing/null identities are ignored, but a malformed or contradictory present value
    makes the run unreadable instead of allowing priority order to hide provenance drift.
    """
    run_dir = Path(run_dir)
    identities: List[int] = []
    for fname in (prov_name, config_name):
        path = run_dir / fname
        if not path.exists():
            continue
        if not path.is_file():
            return None
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(artifact, dict):
            return None
        candidates: List[object] = []
        if "seed" in artifact:
            candidates.append(artifact["seed"])
        if fname == config_name and "config" in artifact:
            nested = artifact["config"]
            if not isinstance(nested, Mapping):
                return None
            if "seed" in nested:
                candidates.append(nested["seed"])
        for candidate in candidates:
            if candidate is None:
                continue
            if not _is_exact_nonnegative_seed(candidate):
                return None
            identities.append(candidate)
    name_match = re.search(r"_s(\d+)(?:_\d+)?$", run_dir.name)
    if name_match is not None:
        identities.append(int(name_match.group(1)))
    if not identities or len(set(identities)) != 1:
        return None
    return identities[0]


def _provenance_contract(run_dir: Path) -> Dict[str, object]:
    """Return the code/data contract required for a comparable across-seed run."""
    path = Path(run_dir) / "provenance.json"
    try:
        provenance = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unreadable provenance at {path}") from exc
    if not isinstance(provenance, Mapping):
        raise ValueError(f"provenance at {path} must be an object")
    seed = provenance.get("seed")
    if not _is_exact_nonnegative_seed(seed):
        raise ValueError(f"provenance at {path} has no exact nonnegative seed")
    git_sha = provenance.get("git_sha")
    git_dirty = provenance.get("git_dirty")
    dirty_fingerprint = provenance.get("git_dirty_fingerprint")
    if not isinstance(git_sha, str) or not git_sha or type(git_dirty) is not bool:
        raise ValueError(f"provenance at {path} has no usable code identity")
    if ((git_dirty and (not isinstance(dirty_fingerprint, str) or not dirty_fingerprint))
            or (not git_dirty and dirty_fingerprint is not None)):
        raise ValueError(f"provenance at {path} has an inconsistent dirty-tree identity")

    contract: Dict[str, object] = {
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "git_dirty_fingerprint": dirty_fingerprint,
    }
    for split in ("train", "val", "test"):
        digest = provenance.get(f"{split}_data_sha256")
        n_tokens = provenance.get(f"{split}_data_n_tokens")
        if (not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None
                or type(n_tokens) is not int
                or n_tokens < 0):
            raise ValueError(f"provenance at {path} has no usable {split} data identity")
        contract[f"{split}_data_sha256"] = digest.lower()
        contract[f"{split}_data_n_tokens"] = n_tokens

    data_seed = provenance.get("data_seed")
    max_tokens = provenance.get("max_tokens")
    tokenizer_tag = provenance.get("tokenizer_tag")
    if data_seed is not None and not _is_exact_nonnegative_seed(data_seed):
        raise ValueError(f"provenance at {path} has an invalid data_seed")
    if max_tokens is not None and (type(max_tokens) is not int or max_tokens <= 0):
        raise ValueError(f"provenance at {path} has an invalid max_tokens")
    if not isinstance(tokenizer_tag, str) or not tokenizer_tag:
        raise ValueError(f"provenance at {path} has no tokenizer identity")
    contract.update({
        "data_seed": data_seed,
        "max_tokens": max_tokens,
        "tokenizer_tag": tokenizer_tag,
    })
    return contract


def _seed_dirs(root: Path, *, validate_configs: bool = True) -> List[Path]:
    r"""The per-seed run dirs directly under ``root`` (those carrying a run artifact), sorted.
    Skips siblings like ``figures/`` that hold no run JSON.

    ``validate_configs=False`` is the raw observation seam used by requested-panel classification:
    unexpected, duplicate, and unidentified directories must be classified before accepted-run
    homogeneity is checked. Direct callers retain the historic fail-loud config validation.
    """
    root = Path(root)
    if not root.exists():
        return []
    out = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and any((p / f).exists()
                              for f in ("summary.json", "provenance.json", "config.json")):
            out.append(p)
    if validate_configs:
        _assert_homogeneous_configs(out)
    return out


def _request_manifest(root: Path, observed: List[int]) -> Dict[str, Any]:
    request_path = root / "multiseed_request.json"
    request = _read_json(request_path)
    observed_valid = all(_is_exact_nonnegative_seed(seed) for seed in observed)
    observed_seeds = sorted(set(observed)) if observed_valid else []
    if not request_path.is_file():
        return {
            "available": False,
            "request_verified": False,
            "requested_seeds": observed_seeds,
            "manifest_status": "unverifiable",
            "declared_statuses": {},
        }
    if not isinstance(request, Mapping):
        return {
            "available": True,
            "request_verified": False,
            "requested_seeds": observed_seeds,
            "manifest_status": "unverifiable",
            "declared_statuses": {},
        }
    raw = request.get("seeds")
    raw_cells = request.get("cells")
    top_status = request.get("status")
    known_statuses = {
        "pending", "running", "complete", "success", "incomplete", "failed",
        "missing", "duplicate", "nonfinite", "unreadable",
    }
    seeds_valid = (
        isinstance(raw, list)
        and bool(raw)
        and all(_is_exact_nonnegative_seed(seed) for seed in raw)
        and len(set(raw)) == len(raw)
    )
    seeds = list(raw) if seeds_valid else observed_seeds
    header_valid = (
        request.get("schema_version") == 1
        and seeds_valid
        and isinstance(top_status, str)
        and top_status in known_statuses
        and isinstance(raw_cells, list)
        and bool(raw_cells)
    )
    declared: Dict[int, str] = {}
    cells_valid = header_valid
    if header_valid:
        for cell in raw_cells:
            if not isinstance(cell, dict):
                cells_valid = False
                continue
            seed = cell.get("seed")
            status = cell.get("status")
            if (
                not _is_exact_nonnegative_seed(seed)
                or seed not in seeds
                or seed in declared
                or not isinstance(status, str)
                or status not in known_statuses
            ):
                cells_valid = False
                continue
            declared[seed] = status
        cells_valid = cells_valid and set(declared) == set(seeds)
    if not cells_valid:
        return {
            "available": True,
            "request_verified": False,
            "requested_seeds": seeds,
            "manifest_status": "unverifiable",
            "declared_statuses": {},
        }
    return {
        "available": True,
        "request_verified": True,
        "requested_seeds": seeds,
        "manifest_status": top_status,
        "declared_statuses": declared,
    }


def _requested_seeds(root: Path, observed: List[int]) -> tuple[List[int], bool]:
    manifest = _request_manifest(root, observed)
    return manifest["requested_seeds"], manifest["request_verified"]


_CODE_IDENTITY_KEYS = ("git_sha", "git_dirty", "git_dirty_fingerprint")
# The data-order seed is reported through ``_data_order_status`` and the noise-band caption; it is
# not a comparability key, so a per-seed data order (``DATA_SEED=None``, data_seed == seed) still
# aggregates, with the 'nonshared_or_mixed' status recorded.
_DATA_ORDER_KEYS = ("data_seed",)


def _mixed_code_identity_allowed() -> bool:
    r"""``CONFIG["allow_mixed_code_identity"]`` (default False), validated as a bool."""
    allowed = CONFIG.get("allow_mixed_code_identity", False)
    if type(allowed) is not bool:
        raise ValueError(f"CONFIG['allow_mixed_code_identity']={allowed!r}; expected a bool")
    return allowed


def _requested_seed_design(
    root: Path,

    *,
    seed_file: str = "config.json",
) -> Dict[str, Any]:
    """Join one invocation-owned seed request to exactly one run directory per requested seed."""
    run_dirs = _seed_dirs(root, validate_configs=False)
    by_seed: Dict[int, List[Path]] = {}
    unidentified_runs: List[str] = []
    for run_dir in run_dirs:
        seed = _seed_for(run_dir, config_name=seed_file)
        if seed is None:
            unidentified_runs.append(str(run_dir))
        else:
            by_seed.setdefault(seed, []).append(run_dir)
    observed_seeds = sorted(seed for seed, dirs in by_seed.items() for _run_dir in dirs)
    manifest = _request_manifest(root, observed_seeds)
    requested = manifest["requested_seeds"]
    requested_set = set(requested)
    unexpected_seeds = sorted(seed for seed in by_seed if seed not in requested_set)
    duplicate_seeds = sorted(seed for seed, dirs in by_seed.items() if len(dirs) > 1)

    cells: List[Dict[str, Any]] = []
    provenance_contracts: Dict[int, Dict[str, object]] = {}
    for seed in requested:
        matches = by_seed.get(seed, [])
        if not manifest["request_verified"]:
            if manifest["available"]:
                cells.append({"seed": seed, "status": "unreadable", "run_dir": None})
            elif len(matches) == 1:
                cells.append({"seed": seed, "status": "complete", "run_dir": str(matches[0])})
            elif len(matches) > 1:
                cells.append({"seed": seed, "status": "duplicate", "run_dir": None})
            else:
                cells.append({"seed": seed, "status": "missing", "run_dir": None})
            continue
        declared = manifest["declared_statuses"][seed]
        if declared not in {"complete", "success"}:
            cells.append({"seed": seed, "status": declared, "run_dir": None})
        elif not matches:
            cells.append({"seed": seed, "status": "missing", "run_dir": None})
        elif len(matches) > 1:
            cells.append({"seed": seed, "status": "duplicate", "run_dir": None})
        else:
            try:
                contract = _provenance_contract(matches[0])
            except ValueError:
                cells.append({"seed": seed, "status": "unverifiable", "run_dir": str(matches[0])})
            else:
                provenance_contracts[seed] = contract
                cells.append({"seed": seed, "status": "complete", "run_dir": str(matches[0])})
    mixed_code_allowed = _mixed_code_identity_allowed()
    code_identities = {
        seed: {key: contract[key] for key in _CODE_IDENTITY_KEYS}
        for seed, contract in provenance_contracts.items()
    }
    n_code_identities = len({
        json.dumps(identity, sort_keys=True) for identity in code_identities.values()
    })
    fingerprints = {
        json.dumps(
            {key: value for key, value in contract.items()
             if key not in _DATA_ORDER_KEYS
             and not (mixed_code_allowed and key in _CODE_IDENTITY_KEYS)},
            sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        )
        for contract in provenance_contracts.values()
    }
    data_order = _data_order_status([
        contract.get("data_seed") for contract in provenance_contracts.values()
    ])
    provenance_verified = (
        bool(cells)
        and len(provenance_contracts) == len(cells)
        and len(fingerprints) == 1
    )
    if manifest["request_verified"] and len(fingerprints) > 1:
        for cell in cells:
            if cell["status"] == "complete":
                cell["status"] = "unverifiable"
    accepted_cells = [
        cell
        for cell in cells
        if cell["status"] == "complete" and cell.get("run_dir") is not None
    ]
    # Configuration homogeneity applies only to the cohort still eligible for aggregation after
    # manifest and provenance classification. Inferred/no-manifest cells remain ``complete`` here,
    # preserving their historical fail-loud homogeneity check.
    _assert_homogeneous_configs([Path(cell["run_dir"]) for cell in accepted_cells])
    accepted_seeds = sorted(
        cell["seed"] for cell in accepted_cells
    )
    exact_observed_panel = (
        accepted_seeds == sorted(requested)
        and not unexpected_seeds
        and not duplicate_seeds
        and not unidentified_runs
    )
    complete = (
        manifest["request_verified"]
        and manifest["manifest_status"] in {"complete", "success"}
        and provenance_verified
        and bool(cells)
        and all(cell["status"] == "complete" for cell in cells)
        and exact_observed_panel
    )
    return {
        "requested_seeds": requested,
        "accepted_seeds": accepted_seeds,
        "observed_seeds": observed_seeds,
        "unexpected_seeds": unexpected_seeds,
        "duplicate_seeds": duplicate_seeds,
        "unidentified_runs": sorted(unidentified_runs),
        "n_accepted_seeds": len(accepted_seeds),
        "n_observed_runs": len(run_dirs),
        "request_verified": manifest["request_verified"],
        "manifest_status": manifest["manifest_status"],
        "provenance_verified": provenance_verified,
        "data_order": data_order,
        "code_identity": {
            "mixed_allowed": mixed_code_allowed,
            "n_distinct": n_code_identities,
            "mixed": n_code_identities > 1,
            "per_seed": {str(seed): identity for seed, identity in sorted(code_identities.items())},
        },
        "status": "complete" if complete else "incomplete",
        "complete": complete,
        "cells": cells,
    }


def _aggregate_requested_metric(
    root: Path,
    key: str,

    *,
    sources: tuple,
    seed_file: str = "config.json",
) -> Dict[str, Any]:
    design = _requested_seed_design(root, seed_file=seed_file)
    values: List[float] = []
    value_seeds: List[int] = []
    cells: List[Dict[str, Any]] = []
    for design_cell in design["cells"]:
        seed = design_cell["seed"]
        if design_cell["status"] != "complete" or design_cell["run_dir"] is None:
            cells.append({"seed": seed, "status": design_cell["status"], "value": None})
            continue
        run_dir = Path(design_cell["run_dir"])
        status = "missing"
        value: Optional[float] = None
        for source in sources:
            source_path = run_dir / source
            if not source_path.is_file():
                continue
            try:
                payload = json.loads(source_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                status = "unreadable"
                break
            if not isinstance(payload, dict):
                status = "unreadable"
                break
            present, raw = _dig_present(payload, key)
            if not present:
                continue
            value = _as_finite_float(raw)
            if value is None:
                status = "nonfinite"
                break
            status = "complete"
            break
        if value is not None:
            values.append(value)
            value_seeds.append(seed)
            cells.append({"seed": seed, "status": "complete", "value": value,
                          "run_dir": str(run_dir)})
        else:
            cells.append({"seed": seed, "status": status, "value": None,
                          "run_dir": str(run_dir)})
    out = _summarize(values)
    out.update({
        "seeds": value_seeds,
        "requested_seeds": design["requested_seeds"],
        "accepted_seeds": design["accepted_seeds"],
        "observed_seeds": design["observed_seeds"],
        "unexpected_seeds": design["unexpected_seeds"],
        "duplicate_seeds": design["duplicate_seeds"],
        "unidentified_runs": design["unidentified_runs"],
        "n_accepted_seeds": design["n_accepted_seeds"],
        "n_observed_runs": design["n_observed_runs"],
        "request_verified": design["request_verified"],
        "manifest_status": design["manifest_status"],
        "cells": cells,
        "complete": (
            design["complete"]
            and bool(cells)
            and all(cell["status"] == "complete" for cell in cells)
        ),
    })
    return out


def _dig_present(d: Dict[str, Any], dotted: str) -> tuple[bool, Any]:
    r"""Nested lookup by dotted key, distinguishing an absent key from an explicit null."""
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False, None
        cur = cur[part]
    return True, cur


def aggregate_scalar(
    run_root: Any,
    key:      str,

    *,
    sources: tuple = ("summary.json", "test_results.json", "research.json"),
) -> Dict[str, Any]:
    r"""Across-seed summary of one (possibly dotted) scalar ``key``, searching ``sources`` in order
    per seed dir. Same return shape as :func:`aggregate_seed_metric`. Handles ``research.json`` keys
    (ECE, ``corpus_freq_strata_ce.rare``, ...) that never appear in ``summary.json``.
    """
    root = _resolve_run_root(run_root)
    return _aggregate_requested_metric(root, key, sources=sources)


def _read_csv_columns(path: Path) -> Dict[str, List[Optional[float]]]:
    r"""Read a CSV into ``{column: [float | None]}``; empty / non-numeric / non-finite cells -> None."""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        cols = list(reader.fieldnames or [])
        data: Dict[str, List[Optional[float]]] = {c: [] for c in cols}
        for row in reader:
            for c in cols:
                data[c].append(_as_finite_float(row.get(c)))
    if ("inner_alignment_energy_total" not in data
            and "free_energy_total" in data):
        data["inner_alignment_energy_total"] = list(data["free_energy_total"])
    return data


def _nan_mean_sd(M: np.ndarray) -> tuple:
    r"""NaN-aware across-row (axis 0) mean, UNBIASED (ddof=1) SD, and finite count per column.
    SD is NaN where fewer than 2 seeds reported; mean is NaN where none did."""
    n = np.sum(np.isfinite(M), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, np.nansum(M, axis=0) / n, np.nan)
        ss = np.nansum((M - mean) ** 2, axis=0)
        sd = np.where(n >= 2, np.sqrt(ss / np.where(n >= 2, n - 1, 1)), np.nan)
    return mean, sd, n


def aggregate_seed_curves(
    run_root: Any,

    *,
    columns:  Optional[List[str]] = None,
    filename: str = "metrics.csv",
    x:        str = "step",
    withheld: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    r"""Across-seed per-step curves from each seed's ``metrics.csv``.

    Returns ``{column: {steps, mean, sd, n}}``. Each curve uses the subset of the shared ``x`` grid
    where at least one seed records that metric; every requested seed must provide a finite aligned
    observation at each such point or the curve set is withheld. This excludes rows that are
    structurally blank for every seed at another metric's cadence. ``columns=None`` selects every
    supported numeric column except ``x`` and ignores columns that are blank for all seeds.

    ``withheld`` (a dict the caller owns) switches to per-column withholding: a column that is
    absent for every seed, or not finite for every seed at its active steps, is left out and
    recorded as ``withheld[column] = reason`` instead of emptying the whole curve set.
    """
    root = _resolve_run_root(run_root)
    design = _requested_seed_design(root)
    if not design["complete"]:
        return {}
    per_seed = []                                                # list of (steps[np], {col: vals[np]})
    all_cols: set = set()
    for cell in design["cells"]:
        run_dir = Path(cell["run_dir"])
        f = run_dir / filename
        if not f.is_file():
            return {}
        try:
            data = _read_csv_columns(f)
        except (OSError, UnicodeError, csv.Error):
            return {}
        if x not in data or not data[x]:
            return {}
        steps = np.array([np.nan if v is None else v for v in data[x]], float)
        keep = np.isfinite(steps)
        steps = steps[keep]
        if not steps.size:
            return {}
        cols = {}
        for c, vals in data.items():
            if c == x:
                continue
            cols[c] = np.array([np.nan if v is None else v for v in vals], float)[keep]
            all_cols.add(c)
        n_repeated = int(steps.size - np.unique(steps).size)
        if n_repeated:
            logger.warning("%s has %d repeated %s value(s); keeping the last finite value per column",
                           f, n_repeated, x)
        per_seed.append((steps, cols))
    if not per_seed:
        return {}
    sel = list(columns) if columns is not None else sorted(all_cols)
    grid = np.unique(np.concatenate([s for s, _ in per_seed]))
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for c in sel:
        stack = []
        for steps, cols in per_seed:
            row = np.full(grid.shape, np.nan)
            if c in cols:
                # Finite values only, in file order, so a repeated step keeps its last finite value.
                finite = np.isfinite(cols[c])
                for idx, value in zip(np.searchsorted(grid, steps[finite]), cols[c][finite]):
                    row[idx] = value
            stack.append(row)
        matrix = np.vstack(stack)
        active = np.any(np.isfinite(matrix), axis=0)
        if not np.any(active):
            if withheld is not None:
                if columns is not None:
                    withheld[c] = "absent or blank for every seed"
                continue
            if columns is not None:
                return {}
            continue
        matrix = matrix[:, active]
        if not np.all(np.isfinite(matrix)):
            if withheld is not None:
                n_bad = int(np.sum(~np.all(np.isfinite(matrix), axis=0)))
                withheld[c] = (f"missing or non-finite for at least one seed at {n_bad} of "
                               f"{matrix.shape[1]} logged {x} values")
                continue
            return {}
        mean, sd, n = _nan_mean_sd(matrix)
        out[c] = {"steps": grid[active], "mean": mean, "sd": sd, "n": n}
    return out


def aggregate_per_layer(
    run_root: Any,

    *,
    filename: str = "metrics_per_layer.csv",
    layer_col: str = "layer",
    withheld: Optional[Dict[str, str]] = None,
) -> Dict[int, Dict[str, Dict[str, Any]]]:
    r"""Across-seed per-layer diagnostics from each seed's ``metrics_per_layer.csv``.
    Returns ``{layer: {column: {mean, sd, n, values}}}`` with the ddof=1 SD across seeds.

    A missing file, layer index or layer set mismatch withholds the whole channel. A column is withheld on
    its own, with its reason in ``withheld`` when given, when it is absent from a seed or not a finite
    number at some (seed, layer): a text column such as ``relational_semantics``, or a column NaN on the
    route (audit 2026-10-10 round 2, #24).
    """
    root = _resolve_run_root(run_root)
    design = _requested_seed_design(root)
    if not design["complete"]:
        return {}
    per_seed: List[Dict[tuple[int, str], Optional[float]]] = []
    seed_layers: List[set] = []
    for cell in design["cells"]:
        run_dir = Path(cell["run_dir"])
        f = run_dir / filename
        if not f.is_file():
            return {}
        try:
            data = _read_csv_columns(f)
        except (OSError, UnicodeError, csv.Error):
            return {}
        if layer_col not in data or not data[layer_col]:
            return {}
        seed_values: Dict[tuple[int, str], Optional[float]] = {}
        for i, lay in enumerate(data[layer_col]):
            if lay is None:
                return {}
            L = int(lay)
            for c, vals in data.items():
                if c != layer_col:
                    seed_values[(L, c)] = vals[i]                   # None: empty, text or non-finite
        if not seed_values:
            return {}
        per_seed.append(seed_values)
        seed_layers.append({L for L, _ in seed_values})
    if any(layers != seed_layers[0] for layers in seed_layers[1:]):
        return {}
    layers = sorted(seed_layers[0])
    columns = sorted({c for seed_values in per_seed for _, c in seed_values})
    keys = []
    for c in columns:
        cells = [seed_values.get((L, c)) for seed_values in per_seed for L in layers]
        if any((L, c) not in seed_values for seed_values in per_seed for L in layers):
            reason = "absent from at least one seed's metrics_per_layer.csv"
        elif all(value is None for value in cells):
            reason = "no finite number for any seed or layer (a text column, or not defined on this route)"
        elif any(value is None for value in cells):
            n_bad = sum(any(seed_values[(L, c)] is None for seed_values in per_seed) for L in layers)
            reason = f"missing or non-finite for at least one seed at {n_bad} of {len(layers)} layer(s)"
        else:
            keys += [(L, c) for L in layers]
            continue
        if withheld is not None:
            withheld[c] = reason
    out: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for L, c in sorted(keys):
        vals = [seed_values[(L, c)] for seed_values in per_seed]
        s = _summarize(vals)
        out.setdefault(L, {})[c] = {
            "mean": s["mean"], "sd": s["sd"], "n": s["n"], "values": vals,
        }
    return out


# =============================================================================
# CLICK-TO-RUN  -- edit, then `python multiseed_analysis.py`.
# =============================================================================
CONFIG: Dict[str, Any] = {
    "run_root": "K=60_GL(10)",   # run folder (bare name resolves under vfe3_runs/) OR a path
    "key":      "best_val_ppl",  # exploratory selection/noise metric; test remains final endpoint
    "curve_x":  "step",          # x axis of the per-step curves: "step" or "tokens_seen" (CURVE_X_OPTIONS)
    # Opt-in: accept seeds whose code identity (git_sha / dirty-tree fingerprint) differs. Data,
    # tokenizer and data-order identity must still match. The across-seed SD then also contains any
    # effect of the code differences; every output records the per-seed code identity.
    "allow_mixed_code_identity": False,
}

# metrics.csv columns accepted as the curve x axis, with their axis labels. ``tokens_seen`` (training
# token presentations) is logged by runs trained on or after 2026-09-28; older runs have only ``step``.
CURVE_X_OPTIONS: Dict[str, str] = {"step": "training step", "tokens_seen": "training tokens"}

# Headline scalars to aggregate (searched in summary.json -> test_results.json -> research.json).
# Dotted keys dig into nested research blocks. ``test_best_*`` are the test split at the
# best-validation weights and ``train_sample_ce_{final,best}`` the fixed training sample under the
# evaluation protocol (summary.json of runs trained on or after 2026-09-28).
SCALAR_KEYS: List[str] = [
    "test_ppl", "best_val_ppl", "test_ce", "test_bits_per_token", "test_bpc",
    "test_best_ce", "test_best_ppl", "test_best_step",
    "train_sample_ce_final", "train_sample_ce_best",
    "wall_time_s", "ece", "overall_ce", "sigma_trace_cv", "sigma_ce_spearman",
    "fd_gradient_worst_rel_error",
    "corpus_freq_strata_ce.rare", "corpus_freq_strata_ce.mid", "corpus_freq_strata_ce.frequent",
]

# Per-step curves to draw an across-seed band for: (metrics.csv column, log-y?). Axis/title labels
# are publication-quality math, resolved centrally by vfe3.viz.figures.pub_label (PUB_LABELS).
CURVE_SPECS: List[tuple] = [
    ("train_ce",           False),
    ("val_ppl",            False),
    ("val_bits_per_token", False),
    ("val_bpc",            False),
    ("inner_alignment_energy_total", False),
    ("self_coupling",      False),
    ("belief_coupling",    False),
    ("attention_entropy",  False),
    ("self_divergence",    False),
    ("grad_norm",          True),
    ("holonomy_deviation", True),
    ("gauge_trace_spread", False),
    ("effective_rank",     False),
    ("attn_entropy",       False),
    ("belief_cond_median", True),
    ("fisher_trace_mean",  False),
    ("generalization_gap", False),
    ("train_sample_ce",    False),                     # fixed training sample, evaluation protocol
    ("generalization_gap_fixed", False),               # val_ce - train_sample_ce
]

# Scalars on a positive ratio scale (losses, perplexities, bits, wall time), the only ones whose
# coefficient of variation SD/|mean| is a seed-stability measure. Signed or near-zero quantities
# (a correlation, a gradient-check error, a step index, ECE) keep their SD on their own
# scale but get no CV and are left out of the seed-stability ranking.
CV_RATIO_SCALE_KEYS = frozenset({
    "test_ppl", "best_val_ppl", "test_ce", "test_bits_per_token", "test_bpc",
    "test_best_ce", "test_best_ppl", "train_sample_ce_final", "train_sample_ce_best",
    "wall_time_s", "overall_ce",
    "corpus_freq_strata_ce.rare", "corpus_freq_strata_ce.mid", "corpus_freq_strata_ce.frequent",
})


def _ratio_scale_cv(key: str, aggregate: Dict[str, Any]) -> Dict[str, Any]:
    r"""Return ``aggregate`` with ``cv`` kept only for a positive ratio-scale metric.

    The CV :math:`s/|\bar x|` is scale-free only when every value is positive on a ratio scale;
    for a signed or near-zero metric it measures the distance of the mean from zero, not the
    across-seed spread, so ``cv`` is set to NaN and ``cv_withheld`` records why.
    """
    out = dict(aggregate)
    if key not in CV_RATIO_SCALE_KEYS:
        out["cv"] = math.nan
        out["cv_withheld"] = "not a positive ratio-scale metric; SD is reported on its own scale"
    elif not all(float(v) > 0.0 for v in out.get("values", [])):
        out["cv"] = math.nan
        out["cv_withheld"] = "a seed value is not positive"
    return out


GRID_COLS: List[str] = [                                          # overview-grid subset
    "train_ce", "val_ppl", "inner_alignment_energy_total", "grad_norm", "holonomy_deviation",
    "gauge_trace_spread", "effective_rank", "attn_entropy", "belief_cond_median",
]
_LOGY_COLS = {c for c, logy in CURVE_SPECS if logy}

PER_LAYER_METRICS: List[str] = [                                  # per-layer bars to draw
    "self_coupling", "belief_coupling", "attention_entropy", "effective_rank",
    "holonomy_deviation", "gauge_trace_spread", "belief_cond_median",
    "self_divergence", "total", "canonical_beta_tv", "canonical_total_delta",   # diagnose_canonical_free_energy
]

def _data_order_status(data_seeds: Iterable[object]) -> Dict[str, object]:
    values = list(data_seeds)
    if values and all(type(value) is int for value in values) and len(set(values)) == 1:
        return {"status": "shared_explicit", "shared": True, "data_seed": values[0]}
    if values and all(value is None for value in values):
        return {"status": "nonshared_unspecified", "shared": False, "data_seed": None}
    return {"status": "nonshared_or_mixed", "shared": False, "data_seed": None}


CAVEAT = ("Data-order sharing is reported only for one explicit common data_seed; data_seed=None "
          "means nonshared/unspecified order, not a shared shuffle stream.")


def _curve_x() -> str:
    r"""The configured curve x column (``CONFIG["curve_x"]``, default ``step``), validated."""
    x = CONFIG.get("curve_x", "step")
    if x not in CURVE_X_OPTIONS:
        raise ValueError(f"CONFIG['curve_x']={x!r}; expected one of {sorted(CURVE_X_OPTIONS)}")
    return x


def _slug(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", s)


def _fnum(x: Any, fmt: str = ".4f") -> str:
    r"""Format a float for a table, or 'n/a' if None / non-finite."""
    return format(x, fmt) if isinstance(x, (int, float)) and math.isfinite(x) else "n/a"


def _json_clean(obj: Any) -> Any:
    r"""Recursively make ``obj`` JSON-valid: numpy scalars/arrays -> python; non-finite floats -> None."""
    if isinstance(obj, dict):
        return {k: _json_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_json_clean(v) for v in obj.tolist()]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    return obj


def _final_finite(d: Dict[str, np.ndarray]) -> Dict[str, Any]:
    r"""Last-finite-step {step, mean, sd, n} of one aggregated curve (the converged value)."""
    finite = np.where(np.isfinite(d["mean"]))[0]
    if finite.size == 0:
        return {"step": None, "mean": None, "sd": None, "n": 0}
    i = int(finite[-1])
    sd = float(d["sd"][i])
    return {"step": float(d["steps"][i]), "mean": float(d["mean"][i]),
            "sd": (sd if math.isfinite(sd) else None), "n": int(d["n"][i])}


def _write_csv(path: Path, scalars: Dict[str, Any]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "n", "mean", "sd", "noise_floor", "cv"])
        for k, v in scalars.items():
            row = [k, v["n"]]
            for key in ("mean", "sd", "noise_floor", "cv"):
                x = v[key]
                row.append("" if not (isinstance(x, (int, float)) and math.isfinite(x)) else x)
            w.writerow(row)


def _write_md(path: Path, root: Path, seeds, scalars, per_layer, *,
              code_identity: Optional[Mapping[str, Any]] = None) -> None:
    L: List[str] = [f"# Multi-seed digest: {root.name}", "",
                    f"Across {len(seeds)} seeds: {seeds}.", ""]
    if code_identity is not None and code_identity.get("mixed"):
        L += [f"**Mixed code identity:** {code_identity['n_distinct']} distinct git_sha / "
              "dirty-tree fingerprints across seeds, accepted by `allow_mixed_code_identity=True`. "
              "The across-seed SD includes any effect of those code differences.", ""]
        for seed, ident in code_identity["per_seed"].items():
            L.append(f"- seed {seed}: `{ident['git_sha'][:10]}` dirty={ident['git_dirty']} "
                     f"fingerprint=`{(ident['git_dirty_fingerprint'] or '-')[:12]}`")
        L.append("")
    L += ["## Headline scalars (across-seed)", "",
          "| metric | n | mean | SD (ddof=1) | noise floor, t(n-1, 0.975)*sqrt(2)*SD | CV% |",
          "|---|---|---|---|---|---|"]
    for k, v in scalars.items():
        cvpct = 100 * v["cv"] if math.isfinite(v["cv"]) else float("nan")
        L.append(f"| {k} | {v['n']} | {_fnum(v['mean'])} | {_fnum(v['sd'])} | "
                 f"{_fnum(v['noise_floor'])} | {_fnum(cvpct, '.3f')} |")
    L += ["", "The noise floor is the 95% half-width for the difference of two single-seed runs; "
          "a single-seed ablation difference below it is within seed noise."]
    finite = {k: v for k, v in scalars.items() if math.isfinite(v["cv"])}
    if finite:
        most = min(finite.items(), key=lambda kv: kv[1]["cv"])
        least = max(finite.items(), key=lambda kv: kv[1]["cv"])
        L += ["", "## Key findings", ""]
        if "test_ppl" in scalars:
            t = scalars["test_ppl"]
            vals = [round(x, 3) for x in t["values"]]
            L.append(f"- test PPL = {_fnum(t['mean'], '.3f')} +/- {_fnum(t['sd'], '.3f')} "
                     f"(single-seed noise floor = {_fnum(t['noise_floor'], '.3f')}); per-seed {vals}.")
        L.append(f"- Most seed-stable ratio-scale metric: `{most[0]}` (CV {100 * most[1]['cv']:.3f}%).")
        L.append(f"- Least seed-stable ratio-scale metric: `{least[0]}` "
                 f"(CV {100 * least[1]['cv']:.3f}%). Signed or near-zero metrics carry no CV; "
                 "read their SD on their own scale.")
    if per_layer:
        L += ["", f"## Per-layer ({len(per_layer)} layer(s))", "",
              "See `figures/per_layer__*.png` for across-seed per-layer bars."]
    L += ["", "## Caveat", "", CAVEAT, "",
          "## Figures", "",
          "In `figures/`: per-metric `curve_band__*.png` + `curve_band_grid.png` (per-step bands), "
          "`scalar_cv_summary.png` (seed stability), `per_layer__*.png`, and "
          f"`noise_band__{_slug(CONFIG['key'])}.png`.", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def _emit_figures(root: Path, scalars, curves, per_layer,
                  data_order: Optional[Mapping[str, Any]] = None) -> None:
    r"""Write the full figure set into ``root/figures``; each figure is best-effort (a plotting /
    dependency error is logged and skipped), mirroring the rest of the figure suite. ``data_order``
    (the request design's ``data_order`` record) sets the noise-band caption."""
    try:
        from vfe3.viz import figures as figs
    except Exception as exc:                                      # matplotlib optional / headless
        print(f"  (figures skipped, plotting unavailable: {exc})")
        return
    figs.set_publication_style()
    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    made = 0
    x_label = CURVE_X_OPTIONS[_curve_x()]

    def _try(make, out: Path) -> None:
        nonlocal made
        try:
            figs.plt.close(make())
            made += 1
        except Exception as exc:                                  # one bad figure never aborts the rest
            print(f"  (figure {out.name} skipped: {exc})")

    def _relabeled(fig, out: Path, *, grid: bool = False):
        r"""Set a curve figure's x label to the configured axis, then save (the builders say 'step')."""
        if x_label != CURVE_X_OPTIONS["step"]:
            if grid:
                fig.supxlabel(x_label, fontsize=9)
            else:
                for ax in fig.axes:
                    ax.set_xlabel(x_label)
        return figs._save(fig, str(out))

    key = CONFIG["key"]
    if key in scalars:
        out = fig_dir / f"noise_band__{_slug(key)}.png"
        order_status = data_order.get("status") if isinstance(data_order, Mapping) else None
        _try(lambda: figs.plot_ppl_noise_band(scalars[key], label=key,
                                              data_order_status=order_status, path=str(out)), out)

    out = fig_dir / "scalar_cv_summary.png"
    _try(lambda: figs.plot_scalar_cv_summary(scalars, path=str(out)), out)

    for col, logy in CURVE_SPECS:
        d = curves.get(col)
        if d is None or not np.any(np.isfinite(d["mean"])):
            continue
        out = fig_dir / f"curve_band__{_slug(col)}.png"
        _try(lambda d=d, logy=logy, col=col, out=out: _relabeled(
             figs.plot_curve_band(d["steps"], d["mean"], d["sd"], label=col,
                                  n=d["n"], logy=logy), out), out)

    grid = [{"steps": curves[c]["steps"], "mean": curves[c]["mean"], "sd": curves[c]["sd"],
             "title": c, "logy": c in _LOGY_COLS}
            for c in GRID_COLS if c in curves and np.any(np.isfinite(curves[c]["mean"]))]
    if grid:
        out = fig_dir / "curve_band_grid.png"
        _try(lambda: _relabeled(figs.plot_curve_band_grid(grid), out, grid=True), out)

    for metric in PER_LAYER_METRICS:
        if per_layer and any(metric in cols for cols in per_layer.values()):
            out = fig_dir / f"per_layer__{_slug(metric)}.png"
            _try(lambda metric=metric, out=out:
                 figs.plot_per_layer_band(per_layer, metric, path=str(out)), out)

    print(f"  figures -> {fig_dir}  ({made} written)")


def main() -> int:
    root = _resolve_run_root(CONFIG["run_root"])
    seed_dirs = _seed_dirs(root, validate_configs=False)
    if not seed_dirs and not (root / "multiseed_request.json").is_file():
        print(f"no per-seed run dirs under {str(root)!r} "
              f"(looked for summary.json / provenance.json / config.json)")
        return 1
    request_design = _requested_seed_design(root)
    seeds = request_design["accepted_seeds"]
    observed_seeds = request_design["observed_seeds"]
    print(
        f"\n=== Multi-seed digest: {root}  "
        f"({len(seeds)} accepted seeds: {seeds}; "
        f"{request_design['n_observed_runs']} observed runs: {observed_seeds}) ===\n"
    )

    scalar_candidates = {
        key: aggregate_scalar(root, key)
        for key in dict.fromkeys([CONFIG["key"], *SCALAR_KEYS])
    }
    headline = scalar_candidates[CONFIG["key"]]
    curve_x = _curve_x()
    withheld_curves: Dict[str, str] = {}
    curves = aggregate_seed_curves(root, x=curve_x, withheld=withheld_curves)
    withheld_per_layer: Dict[str, str] = {}
    per_layer = aggregate_per_layer(root, withheld=withheld_per_layer)
    per_layer_requested = False
    for cell in request_design["cells"]:
        run_dir_value = cell.get("run_dir")
        if run_dir_value is None:
            continue
        run_dir = Path(run_dir_value)
        if (run_dir / "metrics_per_layer.csv").is_file():
            per_layer_requested = True
            break
        try:
            requires_per_layer = _config_requires_per_layer(run_dir / "config.json")
        except ValueError:
            continue
        if requires_per_layer:
            per_layer_requested = True
            break
    per_layer_complete = not per_layer_requested or bool(per_layer)
    partial_scalars = [
        key for key, aggregate in scalar_candidates.items()
        if aggregate["n"] > 0 and not aggregate["complete"]
    ]
    # Scalar publication depends only on the requested design and the headline metric. A scalar
    # that is partial across seeds, or a curve column that one seed leaves blank or non-finite, is
    # withheld on its own and listed with its reason; it never withholds the rest of the digest.
    publication_complete = request_design["complete"] and headline["complete"]
    if publication_complete:
        scalars = {
            key: _ratio_scale_cv(key, aggregate) for key, aggregate in scalar_candidates.items()
            if aggregate["n"] > 0 and aggregate["complete"]
        }
        print(f"{'metric':<28}{'n':>3}{'mean':>14}{'sd':>13}{'floor':>12}{'cv%':>9}")
        print("-" * 79)
        for key, aggregate in scalars.items():
            cvpct = 100 * aggregate["cv"] if math.isfinite(aggregate["cv"]) else float("nan")
            print(f"{key:<28}{aggregate['n']:>3}{aggregate['mean']:>14.4f}"
                  f"{_fnum(aggregate['sd']):>13}{_fnum(aggregate['noise_floor']):>12}"
                  f"{_fnum(cvpct, '.3f'):>9}")
        if not per_layer_complete:
            print("optional per-layer diagnostics are incomplete; that channel is withheld")
        if partial_scalars:
            print(f"withheld scalars (not complete across the accepted seeds): {partial_scalars}")
        if withheld_curves:
            print(f"withheld {len(withheld_curves)} curve column(s); reasons in "
                  "multiseed_summary.json diagnostics.withheld_curves")
    else:
        scalars = {}
        curves = {}
        per_layer = {}
        print("requested multi-seed design is incomplete; aggregate values and figures withheld")
    code_identity = request_design["code_identity"]
    if code_identity["mixed"] and code_identity["mixed_allowed"]:
        print(f"WARNING: {code_identity['n_distinct']} distinct code identities accepted "
              f"(allow_mixed_code_identity=True); the across-seed SD includes any code effect")
    print(f"\nNOTE: {CAVEAT}\n")

    accepted_config_dirs = [
        Path(cell["run_dir"])
        for cell in request_design["cells"]
        if cell["status"] == "complete" and cell.get("run_dir") is not None
    ]
    manifest = {
        "run_root": str(root),
        "config_fingerprint": (
            _config_fingerprint(accepted_config_dirs[0] / "config.json")
            if accepted_config_dirs else None
        ),
        "n_seeds": request_design["n_accepted_seeds"],
        "seeds": seeds,
        "n_accepted_seeds": request_design["n_accepted_seeds"],
        "n_observed_runs": request_design["n_observed_runs"],
        "observed_seeds": observed_seeds,
        "design": {
            "requested_seeds": request_design["requested_seeds"],
            "accepted_seeds": request_design["accepted_seeds"],
            "observed_seeds": request_design["observed_seeds"],
            "unexpected_seeds": request_design["unexpected_seeds"],
            "duplicate_seeds": request_design["duplicate_seeds"],
            "unidentified_runs": request_design["unidentified_runs"],
            "n_accepted_seeds": request_design["n_accepted_seeds"],
            "n_observed_runs": request_design["n_observed_runs"],
            "request_verified": request_design["request_verified"],
            "manifest_status": request_design["manifest_status"],
            "data_order": request_design["data_order"],
            "code_identity": request_design["code_identity"],
            "status": "complete" if publication_complete else "incomplete",
            "complete": publication_complete,
            "cells": headline["cells"],
        },
        "withheld": {
            "scalars": not publication_complete,
            "curves": not publication_complete or not curves,
            "per_layer": not publication_complete or not per_layer_complete,
            "figures": not publication_complete,
        },
        "diagnostics": {
            "headline": headline,
            "partial_scalars": partial_scalars,
            "curves_complete": bool(curves) and not withheld_curves,
            "withheld_curves": withheld_curves,
            "per_layer_requested": per_layer_requested,
            "per_layer_complete": per_layer_complete,
            "withheld_per_layer_columns": withheld_per_layer,
        },
        "caveat": CAVEAT,
        "scalars": scalars,
        "curve_x": curve_x,                                      # the "step" of curves_final_step is this column
        "curves_final_step": {c: _final_finite(d) for c, d in curves.items()},
        "per_layer": {str(L): {c: {"mean": s["mean"], "sd": s["sd"], "n": s["n"]}
                               for c, s in cols.items()}
                      for L, cols in per_layer.items()},
    }
    _write_json_atomic(root / "multiseed_summary.json", _json_clean(manifest))
    _write_csv(root / "multiseed_summary.csv", scalars)
    _write_md(root / "MULTISEED_ANALYSIS.md", root, seeds, scalars, per_layer,
              code_identity=request_design["code_identity"])
    for name in ("multiseed_summary.json", "multiseed_summary.csv", "MULTISEED_ANALYSIS.md"):
        print(f"  data    -> {root / name}")

    if publication_complete:
        _emit_figures(root, scalars, curves, per_layer, request_design["data_order"])
    return 0 if publication_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
