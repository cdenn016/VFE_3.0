r"""Run records for the manuscript assemblers: a declared experiment manifest resolved into runs.

Every cross-run artifact of ``GL(K)_attention_v2`` (Table M2, the per-seed table, Figure M3,
Figure S3, and the single-run appendix tables) is assembled from run directories alone; no model
is built. A manifest declares

* ``arms``: an ordered mapping ``{arm label: runs}``, where ``runs`` is a list of run directories,
  glob patterns, or multi-seed roots (a directory whose children are run directories), or a mapping
  ``{"runs": [...], "group": "<section heading>"}``. An arm with no runs is kept and reported as
  pending, so the manuscript's full row list can be declared before the runs exist;
* ``evaluated_run`` and ``final_run``: the two columns of ``tab:placeholder_resources``; the final
  run also fills ``tab:placeholder_methods_contract`` (the evaluated run when no final run is named);
* ``numerics_run``: the run of Figure S1 (default: the evaluated run);
* ``scaling``: ``{"K": runs, "depth": runs}`` for Figure S3, each optionally
  ``{"runs": [...], "vary": [config fields allowed to differ]}``;
* ``selection``: the checkpoint-selection rule, ``"best"`` (best recorded validation, the default)
  or ``"final"`` (final weights), applied identically to every run;
* ``decision``: optional ``{"evaluated", "comparator", "frozen", "capacity"}`` arm labels of the
  registered decision rule (:func:`vfe3.paper.tables.decision_rule_table`); absent labels take its
  defaults.

An arm mapping may also carry ``"inapplicable": "<reason>"`` (written by ``paper_arms.py`` for a row
that does not apply to its base); the tables then print ``n/a`` for it instead of ``pending``. A
``paper_arms.py`` manifest also holds ``paper_arms.rows[label]``; its ``reuse_verified``, ``stale``, ``failed``
(finished runs without a finite test CE), ``code_identity_mismatch`` and ``window_protocol_mismatch`` records are
carried onto the :class:`Arm` (the decision rule reads them).

:func:`load_run` reads ``config.json`` (strictly, then, for configs written by older code lines,
through :func:`vfe3.legacy_config.lenient_config_from_serialized`, recording the dropped retired
fields), ``summary.json``, ``provenance.json``, ``resources.json`` and ``posthoc/posthoc.json`` when
present. Every quantity is read from these files or derived from the saved config; nothing is
assumed about the run's size, route or history. :func:`check_homogeneity` compares the configs of
the runs of one arm on every field except the model seed (the data-order seed lives in
``provenance.json`` and is not compared), the data and tokenizer identity, and the code identity
(:func:`code_identity`), and reports each differing field. :func:`select_checkpoint`
applies the selection rule to one run and names the source of every number it returns.
"""

from __future__ import annotations

import dataclasses
import glob
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

# Config fields excluded from the within-arm homogeneity check: the model seed (the data-order seed is a
# train_vfe3 argument recorded in provenance.json, not a config field) and the run-cadence fields, which
# decide when a run logs or writes a resumable checkpoint and where it resumes from, not the trained weights
# (a reused base run carries its own cadence; audit 2026-10-17 item 27).
HOMOGENEITY_CADENCE_FIELDS: Tuple[str, ...] = ("checkpoint_interval", "log_interval", "resume_from")
HOMOGENEITY_IGNORED_FIELDS: Tuple[str, ...] = ("seed", *HOMOGENEITY_CADENCE_FIELDS)

# Provenance fields compared across the runs of one arm (data identity and tokenizer).
HOMOGENEITY_PROVENANCE_FIELDS: Tuple[str, ...] = (
    "tokenizer_tag", "max_tokens",
    "train_data_sha256", "train_data_n_tokens",
    "val_data_sha256", "val_data_n_tokens",
    "test_data_sha256", "test_data_n_tokens",
)

# Provenance fields that identify the code a run trained on (as multiseed_analysis._CODE_IDENTITY_KEYS).
CODE_IDENTITY_FIELDS: Tuple[str, ...] = ("git_sha", "git_dirty", "git_dirty_fingerprint")

# Config fields a scaling grid may vary besides the seed, by grid axis (overridable per grid).
SCALING_VARIED_FIELDS: Dict[str, Tuple[str, ...]] = {
    "K": ("embed_dim", "n_heads", "irrep_spec", "group_n"),
    "depth": ("n_layers",),
}
# The config field that is the x coordinate of each scaling axis.
SCALING_AXIS_FIELD: Dict[str, str] = {"K": "embed_dim", "depth": "n_layers"}

SELECTION_RULES: Tuple[str, ...] = ("best", "final")
RUN_MARKERS: Tuple[str, ...] = ("config.json", "summary.json")      # files that identify a run dir
_GLOB_CHARS = re.compile(r"[*?\[]")
_CHECKPOINT_FILE = re.compile(r"^step_(\d+)\.pt$")


# ----------------------------------------------------------------------------------------------
# Records
# ----------------------------------------------------------------------------------------------

@dataclass
class RunRecord:
    r"""The saved facts of one run directory (no model)."""

    run_dir:               Path
    config_record:         Dict[str, Any] = field(default_factory=dict, repr=False)   # config.json
    config_raw:            Dict[str, Any] = field(default_factory=dict, repr=False)   # its "config"
    cfg:                   Any = field(default=None, repr=False)                     # VFE3Config or None
    config_load:           str = "missing"          # "strict" | "lenient" | "failed: ..." | "missing"
    dropped_config_fields: Dict[str, Any] = field(default_factory=dict, repr=False)
    summary:               Dict[str, Any] = field(default_factory=dict, repr=False)
    provenance:            Dict[str, Any] = field(default_factory=dict, repr=False)
    resources:             Optional[Dict[str, Any]] = field(default=None, repr=False)  # resources.json
    posthoc:               Optional[Dict[str, Any]] = field(default=None, repr=False)  # posthoc.json
    seed:                  Optional[int] = None
    data_seed:             Optional[int] = None
    data_seed_recorded:    bool = False             # provenance.json has a data_seed key
    dataset:               Optional[str] = None
    notes:                 List[str] = field(default_factory=list)
    _cache:                Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def name(self) -> str:
        return self.run_dir.name

    def raw(self, key: str, default: Any = None) -> Any:
        r"""The value saved in ``config.json`` for ``key`` (``default`` when the run did not record it)."""
        return self.config_raw.get(key, default)

    def value(self, key: str, default: Any = None) -> Any:
        r"""``key`` from the loaded config when available, else from the raw saved config."""
        if self.cfg is not None and hasattr(self.cfg, key):
            return getattr(self.cfg, key)
        return self.config_raw.get(key, default)

    def cfg_like(self) -> Any:
        r"""The loaded config, or a namespace over the raw saved config when it could not be loaded."""
        return self.cfg if self.cfg is not None else SimpleNamespace(**self.config_raw)

    def metrics(self) -> Optional[Dict[str, Any]]:
        r"""``metrics.csv`` as ``{column: float64 array}`` (cached; None when absent)."""
        if "metrics" not in self._cache:
            from vfe3.viz.figures import read_metrics_columns
            self._cache["metrics"] = read_metrics_columns(self.run_dir / "metrics.csv")
        return self._cache["metrics"]

    def final_step(self) -> Optional[int]:
        r"""The step of the final weights: ``n_steps`` in summary.json (``evaluated_step`` when the
        headline was computed at the final weights), else the last logged step of metrics.csv."""
        candidates = [self.summary.get("n_steps")]
        if self.summary.get("evaluation_weights", "final") == "final" and not self.summary.get("reloaded_best"):
            candidates.append(self.summary.get("evaluated_step"))
        for value in candidates:
            if _is_int(value):
                return int(value)
        metrics = self.metrics()
        if metrics is not None and "step" in metrics:
            steps = [v for v in metrics["step"] if math.isfinite(v)]
            if steps:
                return int(max(steps))
        return None

    def tokens_per_step(self) -> Optional[int]:
        r"""Nominal training tokens per optimizer step, ``batch_size x max_seq_len`` (the training loop
        counts this product per step; gradient accumulation splits the pulled batch)."""
        batch, context = self.raw("batch_size"), self.raw("max_seq_len")
        if _is_int(batch) and _is_int(context):
            return int(batch) * int(context)
        return None


@dataclass
class Arm:
    r"""One declared arm (or scaling grid): its label, section heading, runs and homogeneity record."""

    label:       str
    group:       Optional[str] = None
    runs:        List[RunRecord] = field(default_factory=list)
    unresolved:  List[str] = field(default_factory=list)       # declared entries that matched no run
    homogeneity: Dict[str, Any] = field(default_factory=dict)
    vary:        Tuple[str, ...] = ()
    inapplicable: Optional[str] = None                         # reason the row does not apply (n/a)
    # paper_arms manifest records (paper_arms.rows[label]); None / empty when the manifest has none.
    reuse_verified: Optional[bool] = None                      # False: runs listed without a config check
    stale:        List[Dict[str, Any]] = field(default_factory=list)          # rejected complete runs
    failed:       List[Dict[str, Any]] = field(default_factory=list)  # finished runs without a finite test CE (F61)
    code_identity_mismatch: List[Dict[str, Any]] = field(default_factory=list)   # runs on other code
    # reused runs trained under another training-window protocol (paper_arms audit 2026-10-16 F59)
    window_protocol_mismatch: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Manifest:
    r"""A resolved experiment manifest."""

    arms:      List[Arm] = field(default_factory=list)
    evaluated: Optional[RunRecord] = None
    final:     Optional[RunRecord] = None
    numerics:  Optional[RunRecord] = None
    scaling:   Dict[str, Arm] = field(default_factory=dict)
    selection: str = "best"
    problems:  List[str] = field(default_factory=list)
    decision:  Dict[str, str] = field(default_factory=dict)    # registered decision-rule arm labels


# ----------------------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------------------

def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def finite_float(value: Any) -> Optional[float]:
    r"""``float(value)`` when finite, else None (None, strings, NaN, infinities)."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def is_run_dir(path: Path) -> bool:
    r"""Whether ``path`` is a run directory (holds ``config.json`` or ``summary.json``)."""
    return path.is_dir() and any((path / marker).is_file() for marker in RUN_MARKERS)


def resolve_run_paths(entries: Iterable[Any]) -> Tuple[List[Path], List[str]]:
    r"""Expand declared run entries into run directories; return ``(paths, unresolved entries)``.

    An entry is a run directory, a glob pattern (every matching run directory), or a multi-seed root
    (its immediate child run directories). Order follows the declaration, then the sorted matches;
    duplicates are dropped. A plain entry that matches nothing is looked up by its folder name
    (:func:`find_run_folder`), so a run moved into the archive still resolves; an ambiguous name stays
    unresolved, with the candidates in its text.
    """
    out: List[Path] = []
    unresolved: List[str] = []
    seen = set()

    def add(path: Path) -> None:
        key = str(path.resolve()).lower()
        if key not in seen:
            seen.add(key)
            out.append(path)

    for entry in entries:
        text = str(entry)
        candidates = ([Path(p) for p in sorted(glob.glob(text))] if _GLOB_CHARS.search(text)
                      else [Path(text)])
        matched = False
        for candidate in candidates:
            if is_run_dir(candidate):
                add(candidate)
                matched = True
            elif candidate.is_dir():
                children = [child for child in sorted(candidate.iterdir()) if is_run_dir(child)]
                for child in children:
                    add(child)
                matched = matched or bool(children)
        if not matched and not _GLOB_CHARS.search(text):
            try:
                add(find_run_folder(text))
                matched = True
            except AmbiguousRunFolder as exc:
                text = f"{text} ({exc})"
            except RunFolderNotFound:
                pass
        if not matched:
            unresolved.append(text)
    return out, unresolved


# ----------------------------------------------------------------------------------------------
# Run folders by name
# ----------------------------------------------------------------------------------------------

# Where run folders are searched by name when a declared path no longer exists: the repository's run
# tree and the archive (audit 2026-10-17 item 52). Relative roots resolve against the repository root.
RUN_SEARCH_ROOTS: Tuple[str, ...] = ("vfe3_runs", "D:/DATA_KEEP")
RUN_SEARCH_DEPTH: int = 3                   # root/<name>, root/*/<name>, root/*/*/<name>
_REPO_ROOT = Path(__file__).resolve().parents[2]


class RunFolderNotFound(FileNotFoundError):
    r"""No run folder of the declared name exists at its path or under :data:`RUN_SEARCH_ROOTS`."""


class AmbiguousRunFolder(FileNotFoundError):
    r"""More than one run folder of the declared name exists under :data:`RUN_SEARCH_ROOTS`."""


def find_run_folder(
    entry: Union[str, Path],

    *,
    base:  Optional[Union[str, Path]] = None,
    roots: Optional[Sequence[Union[str, Path]]] = None,
    depth: int = RUN_SEARCH_DEPTH,
) -> Path:
    r"""The run directory ``entry`` names: its path when that is a run directory, else the unique folder
    of the same name under ``roots``.

    A relative ``entry`` resolves against ``base`` (the repository root by default, where every
    click-to-run driver lives), never the working directory. When that path is not a run directory
    (:func:`is_run_dir`), the folder name is searched up to ``depth`` levels below each root of ``roots``
    (:data:`RUN_SEARCH_ROOTS`; relative roots against ``base``), so a run moved into a dated archive folder
    is still found. Raises :class:`RunFolderNotFound` when no run folder of that name exists and
    :class:`AmbiguousRunFolder` when several do (it never chooses between them). Which run a driver names
    is unchanged; only how the name resolves.
    """
    anchor = Path(base) if base is not None else _REPO_ROOT
    path = Path(entry)
    direct = path if path.is_absolute() else anchor / path
    if is_run_dir(direct):
        return direct
    name = path.name
    if not name:
        raise RunFolderNotFound(f"{entry!r} names no folder")
    searched: List[str] = []
    found: Dict[str, Path] = {}
    for root in (RUN_SEARCH_ROOTS if roots is None else roots):
        root_path = Path(root) if Path(root).is_absolute() else anchor / Path(root)
        searched.append(str(root_path))
        if not root_path.is_dir():
            continue
        for level in range(int(depth)):
            pattern = "/".join(["*"] * level + [glob.escape(name)])
            for candidate in root_path.glob(pattern):
                if is_run_dir(candidate):
                    found.setdefault(str(candidate.resolve()).lower(), candidate)
    if len(found) == 1:
        return next(iter(found.values()))
    if not found:
        raise RunFolderNotFound(f"{direct} is not a run directory and no run folder named {name!r} exists "
                                f"under {searched} (depth {depth})")
    raise AmbiguousRunFolder(f"{len(found)} run folders are named {name!r}: "
                             f"{sorted(str(p) for p in found.values())}; name one by its full path")


def load_config(
    config_raw: Mapping[str, Any],

    *,
    source:                     str,
    drop_unknown_config_fields: bool = True,
) -> Tuple[Any, str, Dict[str, Any]]:
    r"""``(cfg or None, load mode, dropped fields)`` for one serialized config.

    Strict :func:`vfe3.config.config_from_serialized` first; when that refuses fields this build no
    longer declares and ``drop_unknown_config_fields`` is set, the lenient loader drops them and
    returns them. A config that still cannot be built yields ``(None, "failed: ...", {})``: the
    assemblers then read the raw saved values only. A config tagged with another ``model_family``
    (the conventional baseline, ``'conventional_transformer'``) is built by that family's loader
    (:func:`vfe3.baseline.baseline_config_from_serialized`), with the same strict-then-lenient rule.
    """
    import warnings

    from vfe3.config import config_from_serialized
    from vfe3.contracts import VFE_MODEL_FAMILY, model_family_of

    family = model_family_of(config_raw)
    if family != VFE_MODEL_FAMILY:
        try:
            from vfe3.baseline import MODEL_FAMILY, baseline_config_from_serialized
            if family != MODEL_FAMILY:
                raise ValueError(f"unknown model_family {family!r}")
            try:
                cfg, _ = baseline_config_from_serialized(dict(config_raw), source=source)
                return cfg, "strict", {}
            except ValueError:
                if not drop_unknown_config_fields:
                    raise
                cfg, dropped = baseline_config_from_serialized(
                    dict(config_raw), source=source, drop_unknown_config_fields=True)
                return cfg, "lenient", dict(dropped)
        except Exception as exc:                              # noqa: BLE001 - any refusal
            return None, f"failed: {type(exc).__name__}: {exc}", {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                       # construction warnings are not ours
        try:
            return config_from_serialized(dict(config_raw), source=source), "strict", {}
        except Exception as strict_exc:                       # noqa: BLE001 - any refusal
            if not drop_unknown_config_fields:
                return None, f"failed: {type(strict_exc).__name__}: {strict_exc}", {}
            try:
                from vfe3.legacy_config import lenient_config_from_serialized
                cfg, dropped = lenient_config_from_serialized(dict(config_raw), source=source)
                return cfg, "lenient", dict(dropped)
            except Exception as exc:                          # noqa: BLE001
                return None, f"failed: {type(exc).__name__}: {exc}", {}


def load_run(
    run_dir: Any,

    *,
    drop_unknown_config_fields: bool = True,
) -> RunRecord:
    r"""Read one run directory into a :class:`RunRecord` (missing files leave empty fields and a note)."""
    run_dir = Path(run_dir)
    record = RunRecord(run_dir=run_dir)
    config_record = _read_json(run_dir / "config.json")
    if config_record is None:
        record.notes.append("config.json absent or unreadable")
    else:
        record.config_record = config_record
        raw = config_record.get("config", config_record)
        record.config_raw = dict(raw) if isinstance(raw, Mapping) else {}
        record.cfg, record.config_load, record.dropped_config_fields = load_config(
            record.config_raw, source=str(run_dir / "config.json"),
            drop_unknown_config_fields=drop_unknown_config_fields)
        if record.config_load.startswith("failed"):
            record.notes.append(f"config could not be built ({record.config_load}); raw values used")
        dataset = config_record.get("dataset")
        record.dataset = dataset if isinstance(dataset, str) else None
    record.summary = _read_json(run_dir / "summary.json") or {}
    if not record.summary:
        record.notes.append("summary.json absent or unreadable")
    record.provenance = _read_json(run_dir / "provenance.json") or {}
    record.resources = _read_json(run_dir / "resources.json")
    record.posthoc = _read_json(run_dir / "posthoc" / "posthoc.json")

    prov_seed, cfg_seed = record.provenance.get("seed"), record.config_raw.get("seed")
    if _is_int(prov_seed):
        record.seed = int(prov_seed)
        if _is_int(cfg_seed) and int(cfg_seed) != record.seed:
            record.notes.append(f"provenance seed {prov_seed} differs from config seed {cfg_seed}")
    elif _is_int(cfg_seed):
        record.seed = int(cfg_seed)
    record.data_seed_recorded = "data_seed" in record.provenance
    data_seed = record.provenance.get("data_seed")
    record.data_seed = int(data_seed) if _is_int(data_seed) else None
    return record


class _RunCache:
    r"""Loads each run directory once, however often a manifest names it."""

    def __init__(self, drop_unknown_config_fields: bool) -> None:
        self.drop = drop_unknown_config_fields
        self.runs: Dict[str, RunRecord] = {}

    def get(self, path: Path) -> RunRecord:
        key = str(path.resolve()).lower()
        if key not in self.runs:
            self.runs[key] = load_run(path, drop_unknown_config_fields=self.drop)
        return self.runs[key]


def _spec_entries(spec: Any) -> Tuple[List[Any], Optional[str], Tuple[str, ...]]:
    r"""``(entries, group, vary)`` of one arm or grid declaration."""
    if spec is None:
        return [], None, ()
    if isinstance(spec, Mapping):
        runs = spec.get("runs") or []
        runs = [runs] if isinstance(runs, (str, Path)) else list(runs)
        return runs, spec.get("group"), tuple(spec.get("vary") or ())
    if isinstance(spec, (str, Path)):
        return [spec], None, ()
    return list(spec), None, ()


def resolve_manifest(
    manifest: Mapping[str, Any],

    *,
    drop_unknown_config_fields: bool = True,
) -> Manifest:
    r"""Resolve a declared manifest (see the module docstring) into run records and homogeneity reports."""
    selection = str(manifest.get("selection", "best"))
    if selection not in SELECTION_RULES:
        raise ValueError(f"selection={selection!r}; expected one of {SELECTION_RULES}")
    cache = _RunCache(drop_unknown_config_fields)
    out = Manifest(selection=selection)
    arms = manifest.get("arms") or {}
    if not isinstance(arms, Mapping):
        raise ValueError("manifest['arms'] must be an ordered mapping {arm label: runs}")
    rows = (manifest.get("paper_arms") or {}).get("rows") or {}
    for label, spec in arms.items():
        entries, group, _ = _spec_entries(spec)
        paths, unresolved = resolve_run_paths(entries)
        arm = Arm(label=str(label), group=group, runs=[cache.get(p) for p in paths],
                  unresolved=unresolved)
        if isinstance(spec, Mapping) and isinstance(spec.get("inapplicable"), str):
            arm.inapplicable = spec["inapplicable"]
        row = rows.get(label)
        if isinstance(row, Mapping):
            if isinstance(row.get("reuse_verified"), bool):
                arm.reuse_verified = row["reuse_verified"]
            arm.stale = list(row.get("stale") or [])
            arm.failed = list(row.get("failed") or [])
            arm.code_identity_mismatch = list(row.get("code_identity_mismatch") or [])
            arm.window_protocol_mismatch = list(row.get("window_protocol_mismatch") or [])
        arm.homogeneity = check_homogeneity(arm.runs)
        for entry in unresolved:
            out.problems.append(f"arm {label!r}: {entry!r} matched no run directory")
        out.arms.append(arm)

    def single(key: str) -> Optional[RunRecord]:
        entry = manifest.get(key)
        if entry in (None, ""):
            return None
        paths, unresolved = resolve_run_paths([entry])
        if unresolved or len(paths) != 1:
            out.problems.append(f"{key}: {entry!r} must name exactly one run directory "
                                f"(matched {len(paths)})")
            return None
        return cache.get(paths[0])

    decision = manifest.get("decision") or {}
    if not isinstance(decision, Mapping):
        raise ValueError("manifest['decision'] must be a mapping of decision-rule arm labels")
    out.decision = {str(key): str(value) for key, value in decision.items() if value is not None}
    out.evaluated = single("evaluated_run")
    out.final = single("final_run")
    out.numerics = single("numerics_run") or out.evaluated
    for axis, spec in (manifest.get("scaling") or {}).items():
        if axis not in SCALING_AXIS_FIELD:
            out.problems.append(f"scaling axis {axis!r} is not one of {sorted(SCALING_AXIS_FIELD)}")
            continue
        entries, _, vary = _spec_entries(spec)
        paths, unresolved = resolve_run_paths(entries)
        grid = Arm(label=axis, runs=[cache.get(p) for p in paths], unresolved=unresolved,
                   vary=vary or SCALING_VARIED_FIELDS[axis])
        grid.homogeneity = check_homogeneity(grid.runs, allowed=grid.vary)
        for entry in unresolved:
            out.problems.append(f"scaling {axis!r}: {entry!r} matched no run directory")
        out.scaling[axis] = grid
    return out


# ----------------------------------------------------------------------------------------------
# Homogeneity
# ----------------------------------------------------------------------------------------------

def code_identity(run: RunRecord) -> Dict[str, Any]:
    r"""The code a run trained on: the source archive's content hash when recorded, else the git triple.

    ``{"source_tree_sha256": ...}`` when ``provenance.json`` holds an available source archive (a
    content hash of the archived files, insensitive to commits that change no archived file), else
    ``{"git_sha", "git_dirty", "git_dirty_fingerprint"}`` (None where unrecorded).
    """
    archive = run.provenance.get("source_archive")
    if isinstance(archive, Mapping) and archive.get("available") and isinstance(archive.get("tree_sha256"), str):
        return {"source_tree_sha256": archive["tree_sha256"]}
    return {key: run.provenance.get(key) for key in CODE_IDENTITY_FIELDS}


def current_code_identity(kind: str) -> Dict[str, Any]:
    r"""The working tree's code identity in the form :func:`code_identity` records.

    ``kind="source_tree_sha256"``: the content hash of the files a source archive would hold
    (:func:`vfe3.run_archive.source_tree_sha256`); any other kind: the git triple of
    ``vfe3.run_artifacts._git_code_identity`` (None where the probe fails).
    """
    if kind == "source_tree_sha256":
        from vfe3.run_archive import source_tree_sha256
        return {"source_tree_sha256": source_tree_sha256()[0]}
    from vfe3.run_artifacts import _git_code_identity
    identity = _git_code_identity()
    return {key: identity.get(key) for key in CODE_IDENTITY_FIELDS}


def code_identity_mismatch(
    run: RunRecord,

    *,
    current: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Optional[str]:
    r"""Why ``run``'s recorded code identity is not the working tree's, or None when it is.

    The run's :func:`code_identity` is compared with :func:`current_code_identity` of the same kind
    (the source-archive content hash when the run recorded one, else the git triple). A run that
    records no identity, or a tree whose identity cannot be probed, is a mismatch with that reason.
    ``current`` caches the tree's identity by kind across calls.
    """
    recorded = code_identity(run)
    if all(value is None for value in recorded.values()):
        return "the run records no code identity"
    kind = "source_tree_sha256" if "source_tree_sha256" in recorded else "git"
    cache = {} if current is None else current
    if kind not in cache:
        cache[kind] = current_code_identity(kind)
    now = cache[kind]
    if all(value is None for value in now.values()):
        return f"the current tree's code identity could not be determined ({now})"
    if _canonical(recorded) != _canonical(now):
        return f"trained on {recorded}, the current tree is {now}"
    return None


def _comparable_fields(run: RunRecord) -> Dict[str, Any]:
    r"""Every config field (normalized through the loaded config when possible), the retired fields
    dropped on reload, the dataset, and the provenance data and tokenizer identity.

    The code identity is not compared here: :func:`check_homogeneity` reports it as a note, never as a
    violation (audit 2026-10-17 item 27)."""
    if run.cfg is not None:
        values: Dict[str, Any] = {f.name: getattr(run.cfg, f.name) for f in dataclasses.fields(run.cfg)}
        values.update({f"retired:{key}": value for key, value in run.dropped_config_fields.items()})
    else:
        values = dict(run.config_raw)
    values["config.json:dataset"] = run.dataset
    for key in HOMOGENEITY_PROVENANCE_FIELDS:
        values[f"provenance:{key}"] = run.provenance.get(key)
    return values


def _integral_floats_as_int(value: Any) -> Any:
    r"""Integral finite floats as ints, recursively, so ``1`` and ``1.0`` compare equal (audit 2026-10-08 general-C#7)."""
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    if isinstance(value, Mapping):
        return {key: _integral_floats_as_int(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_integral_floats_as_int(item) for item in value]
    return value


def _canonical(value: Any) -> str:
    return json.dumps(_integral_floats_as_int(value), sort_keys=True, default=str, allow_nan=True)


def check_homogeneity(
    runs:    Sequence[RunRecord],

    *,
    ignore:  Sequence[str] = HOMOGENEITY_IGNORED_FIELDS,
    allowed: Sequence[str] = (),
) -> Dict[str, Any]:
    r"""Compare the runs of one arm on every config field except ``ignore`` (and ``allowed``).

    Returns ``{"n_runs", "homogeneous", "violations": [{"field", "values": {run: value}}],
    "allowed_differences", "seeds", "data_seeds", "duplicate_seed_pairs", "model_seed_varied",
    "data_seed_varied", "code_identity_differences"}``. A violation is reported, not raised: the
    assemblers still aggregate the arm and mark it. ``allowed`` names fields a scaling grid varies on
    purpose; their differences are listed under ``allowed_differences``.

    The run-cadence fields (:data:`HOMOGENEITY_CADENCE_FIELDS`) are ignored by default. The code identity
    (:func:`code_identity`) is never decisive: every edit to an archived root script changes it, so runs
    whose identities differ are listed under ``code_identity_differences`` as a note and do not make the
    arm non-homogeneous (audit 2026-10-17 item 27).

    The training-window protocol (``train_window_mode``, :func:`vfe3.experiment.saved_train_window_mode`) is
    compared too: the loaders drop a saved ``train_window_offset`` silently, and fixed-window and per-epoch-random
    runs present different training windows, so they are not replicates (audit 2026-10-16 F59). Only known modes
    are compared; a run whose protocol is ``"unknown"`` blocks nothing.
    """
    ignore_set, allowed_set = set(ignore), set(allowed)
    per_run = [(run.name, _comparable_fields(run)) for run in runs]
    keys = sorted({key for _, values in per_run for key in values} - ignore_set)
    violations: List[Dict[str, Any]] = []
    allowed_differences: List[Dict[str, Any]] = []
    for key in keys:
        seen = {name: values.get(key, "<absent>") for name, values in per_run}
        if len({_canonical(v) for v in seen.values()}) > 1:
            (allowed_differences if key in allowed_set else violations).append(
                {"field": key, "values": seen})
    if "train_window_mode" not in ignore_set:
        from vfe3.experiment import saved_train_window_mode

        modes = {run.name: saved_train_window_mode(run.config_raw, run.provenance)[0] for run in runs}
        if len({mode for mode in modes.values() if mode != "unknown"}) > 1:
            (allowed_differences if "train_window_mode" in allowed_set else violations).append(
                {"field": "train_window_mode", "values": modes})
    identities = {run.name: code_identity(run) for run in runs}
    code_differences = ([{"field": "provenance:code_identity", "values": identities}]
                        if len({_canonical(v) for v in identities.values()}) > 1 else [])
    pairs = [(run.seed, run.data_seed) for run in runs]
    duplicates = sorted({pair for pair in pairs if pairs.count(pair) > 1}, key=str)
    seeds = [run.seed for run in runs]
    data_seeds = [run.data_seed for run in runs]
    return {
        "n_runs": len(runs),
        "homogeneous": not violations and not duplicates,
        "violations": violations,
        "allowed_differences": allowed_differences,
        "seeds": seeds,
        "data_seeds": data_seeds,
        "duplicate_seed_pairs": [list(pair) for pair in duplicates],
        "model_seed_varied": len({s for s in seeds if s is not None}) > 1,
        "data_seed_varied": len({s for s in data_seeds if s is not None}) > 1,
        "code_identity_differences": code_differences,
    }


# ----------------------------------------------------------------------------------------------
# Checkpoint selection
# ----------------------------------------------------------------------------------------------

def posthoc_pass(run: RunRecord, checkpoint: str, split: str) -> Optional[Dict[str, Any]]:
    r"""The ``posthoc.json`` pass for ``(checkpoint, split)`` on the FULL split with status ok, or None."""
    for entry in (run.posthoc or {}).get("passes") or []:
        if (entry.get("checkpoint") == checkpoint and entry.get("split") == split
                and entry.get("status") == "ok" and not entry.get("subset")
                and finite_float(entry.get("ce")) is not None):
            return entry
    return None


def _headline_weights(run: RunRecord) -> str:
    r"""Which weights the summary.json headline test was computed at (``"final"`` or ``"best"``)."""
    if run.summary.get("evaluation_weights") == "best" or run.summary.get("reloaded_best") is True:
        return "best"
    return "final"


def select_checkpoint(run: RunRecord, rule: str = "best") -> Dict[str, Any]:
    r"""Apply the checkpoint-selection rule to one run.

    ``"best"``: the weights with the lowest recorded validation perplexity. The selected step is
    ``best_step``; the validation CE is :math:`\log(\mathrm{best\_val\_ppl})`; the test CE is, in
    order, the headline test when the run reloaded the best weights, ``test_best_ce`` (runs trained on
    or after 2026-09-28), the post-hoc full-split test pass at ``best_model.pt``, or the headline test
    when the best step is the final step. ``"final"``: the final weights, ``final_val_ce``, and the
    headline test at the final weights (or the post-hoc pass at ``final_model.pt``).

    Returns ``{"rule", "step", "val_ce", "val_source", "test_ce", "test_ppl", "test_source",
    "status", "reason"}`` with :math:`\mathrm{PPL} = e^{\mathrm{CE}}`; ``status`` is ``"ok"`` when a
    test CE was found, else ``"unavailable"`` with the reason.
    """
    if rule not in SELECTION_RULES:
        raise ValueError(f"rule={rule!r}; expected one of {SELECTION_RULES}")
    s = run.summary
    out: Dict[str, Any] = {"rule": rule, "step": None, "val_ce": None, "val_source": None,
                           "test_ce": None, "test_ppl": None, "test_source": None,
                           "status": "unavailable", "reason": None}
    final_step = run.final_step()
    headline = _headline_weights(run)
    if rule == "best":
        step = s.get("test_best_step") if _is_int(s.get("test_best_step")) else s.get("best_step")
        out["step"] = int(step) if _is_int(step) else None
        best_ppl = finite_float(s.get("best_val_ppl"))
        if best_ppl is not None and best_ppl > 0:
            out["val_ce"], out["val_source"] = math.log(best_ppl), "summary.json log(best_val_ppl)"
        candidates: List[Tuple[Optional[float], str]] = []
        if headline == "best":
            candidates.append((finite_float(s.get("test_ce")), "summary.json test_ce (best weights reloaded)"))
        candidates.append((finite_float(s.get("test_best_ce")), "summary.json test_best_ce"))
        entry = posthoc_pass(run, "best_model.pt", "test")
        candidates.append((finite_float(entry.get("ce")) if entry else None,
                           "posthoc.json best_model.pt test (full split)"))
        if out["step"] is not None and final_step is not None and out["step"] == final_step \
                and headline == "final":
            candidates.append((finite_float(s.get("test_ce")),
                               "summary.json test_ce (best step is the final step)"))
        reason = ("no test at the best-validation weights (summary.json test_best_ce, or "
                  "posthoc_eval.py on best_model.pt)")
    else:
        out["step"] = final_step
        val = finite_float(s.get("final_val_ce"))
        if val is not None:
            out["val_ce"], out["val_source"] = val, "summary.json final_val_ce"
        else:
            entry = posthoc_pass(run, "final_model.pt", "validation")
            if entry is not None:
                out["val_ce"], out["val_source"] = float(entry["ce"]), "posthoc.json final_model.pt validation"
        candidates = []
        if headline == "final":
            candidates.append((finite_float(s.get("test_ce")), "summary.json test_ce"))
        entry = posthoc_pass(run, "final_model.pt", "test")
        candidates.append((finite_float(entry.get("ce")) if entry else None,
                           "posthoc.json final_model.pt test (full split)"))
        reason = "no test at the final weights (summary.json test_ce, or posthoc_eval.py on final_model.pt)"
    for value, source in candidates:
        if value is not None:
            out.update(test_ce=value, test_ppl=math.exp(value), test_source=source, status="ok")
            return out
    out["reason"] = reason
    return out


# ----------------------------------------------------------------------------------------------
# Cost and size records
# ----------------------------------------------------------------------------------------------

def device_hours(run: RunRecord) -> Tuple[Optional[float], Optional[str]]:
    r"""Device-hours of training (one device): ``resources.json device_hours``, else
    ``summary.json wall_time_s / 3600``, else the last ``wall_clock_s`` of metrics.csv / 3600."""
    value = finite_float((run.resources or {}).get("device_hours"))
    if value is not None:
        return value, "resources.json device_hours"
    wall = finite_float(run.summary.get("wall_time_s"))
    if wall is not None:
        return wall / 3600.0, "summary.json wall_time_s"
    metrics = run.metrics()
    if metrics is not None and "wall_clock_s" in metrics:
        finite = [v for v in metrics["wall_clock_s"] if math.isfinite(v)]
        if finite:
            return max(finite) / 3600.0, "metrics.csv wall_clock_s (training loop only)"
    return None, None


def train_device_hours(run: RunRecord) -> Tuple[Optional[float], Optional[str]]:
    r"""Training-only device-hours: :math:`\sum_r \bar t_r \Delta s_r / 3600` over the logging windows.

    The logged mean optimizer-step time times the steps of each window
    (:func:`vfe3.run_resources.throughput_summary`), which excludes evaluation and artifact writing;
    this is the quantity the compute-matched arm matches (:func:`vfe3.baseline.measured_train_device_hours`).
    Falls back to :func:`device_hours` (wall time, evaluation included) with its source when
    ``metrics.csv`` records no step time.
    """
    if "train_device_hours" not in run._cache:
        value = None
        try:
            from vfe3.run_resources import throughput_summary
            value = finite_float(throughput_summary(run.run_dir).get("train_only_hours_estimate"))
        except Exception as exc:                              # noqa: BLE001 - recorded, not raised
            run.notes.append(f"training-only device-hours unavailable: {exc!r}")
        if value is not None and value > 0:
            run._cache["train_device_hours"] = (value, "metrics.csv train_step_ms_mean x steps (training only)")
        else:
            hours, source = device_hours(run)
            run._cache["train_device_hours"] = (
                hours, f"{source} (wall time incl. evaluation; no training-step timing)" if source else None)
    return run._cache["train_device_hours"]


def latest_training_checkpoint(run: RunRecord) -> Optional[Path]:
    r"""The ``checkpoints/step_<N>.pt`` with the largest ``N`` (None when the run kept none)."""
    directory = run.run_dir / "checkpoints"
    if not directory.is_dir():
        return None
    found = [(int(m.group(1)), p) for p in directory.iterdir()
             if (m := _CHECKPOINT_FILE.match(p.name)) is not None]
    return max(found)[1] if found else None


def _parameter_cfg_like(run: RunRecord) -> SimpleNamespace:
    r"""The config fields the parameter breakdown reads (family, vocabulary, decode, frozen tables)."""
    return SimpleNamespace(
        model_family=run.value("model_family", "vfe") or "vfe",
        vocab_size=run.value("vocab_size"),
        decode_mode=run.value("decode_mode"),
        gauge_transport=run.value("gauge_transport", "on"),
        # Historical key: absent on runs after its 2026-10-02 retirement, which froze the temperature.
        decode_temperature_learned=run.value("decode_temperature_learned", False),
    )


def parameter_counts(run: RunRecord, *, read_checkpoints: bool = True) -> Dict[str, Any]:
    r"""Parameter breakdown by role (:func:`vfe3.run_resources.parameter_breakdown`), trainable only.

    Sources in order: ``resources.json``, ``posthoc.json`` (``resources.parameter_breakdown``), the
    breakdown of ``final_model.pt`` read memory-mapped (shapes only), and finally the total alone from
    ``summary.json`` (``n_params_trainable``, else ``n_params``). Returns the breakdown sections plus
    ``source`` (None when nothing is recorded). Tables the config freezes by design
    (``requires_grad=False``: the token and positional frames under ``gauge_transport`` ``'off'`` or
    ``'frozen'``) are excluded from every section and from ``total`` and listed under ``frozen``; a
    breakdown recorded before that section existed is recomputed from its recorded shapes
    (:func:`vfe3.run_resources.exclude_frozen_from_breakdown`).
    """
    if "parameter_counts" in run._cache:
        return run._cache["parameter_counts"]
    breakdown, source = None, None
    for candidate, name in (((run.resources or {}).get("parameter_breakdown"), "resources.json"),
                            ((((run.posthoc or {}).get("resources") or {}).get("parameter_breakdown")),
                             "posthoc.json")):
        if isinstance(candidate, Mapping) and _is_int(candidate.get("total")):
            breakdown, source = candidate, name
            break
    if breakdown is not None:
        try:
            from vfe3.run_resources import exclude_frozen_from_breakdown
            corrected = exclude_frozen_from_breakdown(breakdown, _parameter_cfg_like(run))
            if corrected is not None:
                breakdown = corrected
                source = f"{source} (frozen tables excluded from its recorded shapes)"
        except Exception as exc:                              # noqa: BLE001 - recorded, not raised
            run.notes.append(f"frozen-table correction of the {source} breakdown failed: {exc!r}")
    if breakdown is None and read_checkpoints and (run.run_dir / "final_model.pt").is_file():
        try:
            from vfe3.posthoc.maps import load_map_state
            from vfe3.run_resources import frozen_parameter_names, parameter_breakdown
            state, _ = load_map_state(run.run_dir / "final_model.pt")
            cfg_like = _parameter_cfg_like(run)
            breakdown = parameter_breakdown(state, cfg_like,
                                            frozen=frozen_parameter_names(cfg_like, state.keys()))
            source = "final_model.pt (vfe3.run_resources.parameter_breakdown)"
        except Exception as exc:                              # noqa: BLE001 - recorded, not raised
            run.notes.append(f"parameter breakdown from final_model.pt failed: {exc!r}")
    out: Dict[str, Any] = {"source": source}
    if breakdown is not None:
        out.update({key: value for key, value in breakdown.items() if key != "roles_by_name"})
    elif _is_int(run.summary.get("n_params_trainable")):
        out.update(total=int(run.summary["n_params_trainable"]),
                   source="summary.json n_params_trainable (total only)")
    elif _is_int(run.summary.get("n_params")):
        out.update(total=int(run.summary["n_params"]), source="summary.json n_params (total only)")
    run._cache["parameter_counts"] = out
    return out


def optimizer_groups(run: RunRecord) -> Optional[Dict[str, Any]]:
    r"""The executed optimizer groups from the latest training checkpoint (metadata only, mmap).

    Returns ``{"checkpoint", "step", "groups": [{"names", "role", "aux_role", "initial_lr", "lr",
    "weight_decay", "decoupled_weight_decay", "successful_updates", "table_fisher"}]}`` or None when the
    run kept no training checkpoint (``save_model_artifacts=False`` or deleted) or it cannot be read.
    ``table_fisher`` is the executed marker (``"mu"``/``"sigma"``) of a group the natural-gradient table
    step of :mod:`vfe3.table_optim` moves under ``m_table_update_mode='fisher'``, else None.
    """
    if "optimizer_groups" in run._cache:
        return run._cache["optimizer_groups"]
    path = latest_training_checkpoint(run)
    result = None
    if path is not None:
        try:
            import torch
            bundle = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            groups = (bundle.get("optimizer_state") or {}).get("param_groups") or []
            manifest = bundle.get("optimizer_parameter_manifest") or {}
            names = manifest.get("parameter_groups") if isinstance(manifest, Mapping) else None
            rows = []
            for index, group in enumerate(groups):
                entry_names = ([str(e.get("name")) for e in names[index]]
                               if isinstance(names, list) and index < len(names) else [])
                rows.append({
                    "names": entry_names,
                    "role": group.get("role"),
                    "aux_role": group.get("lr_aux_role"),
                    "initial_lr": finite_float(group.get("initial_lr")),
                    "lr": finite_float(group.get("lr")),
                    "weight_decay": finite_float(group.get("weight_decay")),
                    "decoupled_weight_decay": group.get("decoupled_weight_decay"),
                    "successful_updates": group.get("successful_updates"),
                    "table_fisher": group.get("table_fisher"),
                })
            step = bundle.get("step")
            result = {"checkpoint": str(path), "step": int(step) if _is_int(step) else None,
                      "groups": rows}
        except Exception as exc:                              # noqa: BLE001 - recorded, not raised
            run.notes.append(f"optimizer groups from {path.name} unreadable: {exc!r}")
    run._cache["optimizer_groups"] = result
    return result
