r"""Across-seed aggregation of the semantic-geometry analysis (:mod:`vfe3.semantics`).

Input: two or more trained runs of one configuration that differ in the seed (and possibly in the
training data order), each analyzed by :func:`vfe3.semantics.report.analyze_run` (the ``"quick"``
preset is the intended per-seed battery), so that ``<run>/semantics/metrics.json`` (the click-to-run
analysis), ``<run>/figures/semantics/metrics.json`` (the analysis the training report runs) or a
pre-preset ``<run>/figures/metrics.json`` exists. :func:`aggregate_seeds` writes into ``out_dir``:

* ``multiseed_semantics.json`` -- runs, homogeneity, data order, every headline metric per seed with
  its across-seed summary, and the cross-seed stability matrices;
* ``MULTISEED_SEMANTICS.md`` -- the same as tables, with the data-order caveat at the top;
* ``figures/headline_views.png|pdf`` and ``figures/headline_sections.png|pdf`` -- each headline
  metric per seed with the across-seed mean and :math:`\pm 1` SD;
* ``figures/stability.png|pdf`` -- seed x seed heatmaps of kNN Jaccard@10 and debiased linear CKA
  per channel.

Headline metrics (:func:`headline_metrics`). Per embedding view present in every seed: Google
3CosAdd (query words excluded) and its margin over ONLY-B, micro-averaged over questions and, when
the battery records it, macro-averaged over categories; semantic-field kNN purity@10 with its
label-permutation null mean and z; SimLex-999 and WordSim-353 Spearman :math:`\rho`. Σ: the
coefficient of variation of :math:`\mathrm{tr}\,\Sigma` and the SimLex-999 :math:`\rho` of the
−Jeffreys scorer with its paired-bootstrap :math:`\Delta\rho` over the Σ-free control
:math:`-\lVert\Delta\mu\rVert`. φ: group-analogy chart accuracy, ONLY-B and their margin, and the
positional :math:`R^2` of the offset grouping with its position-permutation null mean. Contextual:
the logit-lens CE at the final layer and the unigram readout, and the final-layer distilled field
purity, 3CosAdd, ONLY-B and margin. Across seeds each metric gets the mean, the unbiased SD
(:math:`n - 1`), the minimum and the maximum over the seeds that report it.

Cross-seed stability (:func:`stability_analysis`). For each channel -- the causal decode view
(``decode_W_causal`` for a linear decode, ``decode_natural_causal`` for a Gaussian KL decode), the
gauge-fixed frame chart ``phi_log`` and the raw prior means ``mu_raw`` -- every seed's rows are
rebuilt from its checkpoint with :func:`~vfe3.semantics.run.load_semantic_run` on the candidate ids
all seeds share (the intersection of each analysis's ``candidate_ids.npy``, in the first seed's
frequency order; rows without a real frame logarithm in any seed are dropped from ``phi_log``). With
:math:`\bar X_s` the column-centered rows of seed :math:`s` (one row per shared word):

* kNN Jaccard@10: :math:`N_s(v)` are the 10 nearest other words of :math:`v` by the cosine of the
  rows of :math:`\bar X_s`, and
  :math:`J_{ab} = \mathrm{mean}_v\, |N_a(v)\cap N_b(v)| / |N_a(v)\cup N_b(v)|`; its chance level is
  the expectation for two independent uniform 10-subsets of the other :math:`n - 1` words
  (hypergeometric);
* linear CKA (Kornblith et al. 2019),
  :math:`\mathrm{CKA}(X_a, X_b) = \lVert\bar X_b^\top\bar X_a\rVert_F^2 /
  (\lVert\bar X_a^\top\bar X_a\rVert_F\,\lVert\bar X_b^\top\bar X_b\rVert_F)`, and its debiased form
  with the unbiased HSIC of Song et al. (2012). The biased form is inflated when :math:`n` is not
  much larger than the widths (about :math:`D/(n + D)` for unrelated isotropic rows), so each pair
  is also scored against a seeded row permutation of the second seed, which destroys the word
  matching and gives the no-relation reference of both forms.

Gauge invariance of the stability metrics. Two seeds are two trivializations as well as two
solutions: each seed's tables are expressed in coordinates that are fixed only up to the model's
redundancies, chosen independently per seed. Both metrics depend on a seed's rows only through the
Gram matrix of its centered rows up to a positive factor, so they are invariant under
:math:`X_s \to c X_s Q + \mathbf 1 t^\top` for any :math:`c > 0`, orthogonal :math:`Q` and
:math:`t`, applied separately to each seed, and neither needs the seeds to share a width or any
alignment of coordinates. The channels are chosen so that each seed's trivialization acts inside
that class. (1) The causal decode rows :math:`Z = (W - \bar W)\,\mathrm{Cov}(W)^{-1/2}`: a global
invertible reparameterization :math:`\mu \to A\mu` compensated in the decoder maps the rows
:math:`w \to A^{-\top} w` and the whitened rows :math:`Z \to Z Q` with :math:`Q` orthogonal, so both
metrics are exactly invariant under it (this includes the global positive-diagonal rescaling and any
coordinate permutation); for the Gaussian decode the natural parameters
:math:`\eta_v = (m_v/t_v, -1/(2t_v))/\tau_\mathrm{eff}` transform linearly under the diagonal
rescaling, with the same conclusion. (2) The frame chart :math:`L_v = \log(D_0 e^{X_v} m^{-1}
D_0^{-1})` is invariant under the split redundancy :math:`e^{X_v} \to e^{X_v} k` and the diagonal
rescaling at the barycenter fixed point (up to the barycenter residual otherwise), so a seed's rows do
not depend on its trivialization at all. (3) The raw prior means change under the diagonal rescaling
as :math:`\mu \to D\mu`, which is outside the invariance class: when that rescaling is an exact or
assumed redundancy of the configuration (:func:`~vfe3.semantics.common.diagonal_redundancy_status`),
the ``mu_raw`` stability depends on each seed's arbitrary diagonal scale; when a LayerNorm breaks it,
the raw coordinates are not a gauge choice. The outputs state which case applies. Neighbor tables
of one seed remain single-seed illustrations (Antoniak and Mimno 2018; Wendlandt et al. 2018).

Homogeneity and data order (:func:`config_homogeneity`, :func:`data_order_status`). The checkpoint
configurations must agree in every field except ``seed``; differing fields are reported, never
silently averaged over. The effective training data seed comes from each run's
``provenance.json``: when every seed shares it, the across-seed SD measures initialization and
optimization only, a lower bound on the variance over data orders. The caveat is written into the
JSON, the markdown and the headline figures.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import math
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from vfe3.semantics.common import (
    OKABE_ITO,
    apply_style,
    diagonal_redundancy_status,
    finite_float as _num,
    markdown_table,
    nested_get as _get,
    save_figure,
    write_json,
)


SEMANTICS_SUBDIRS: Tuple[str, ...] = ("semantics", "figures/semantics", "figures")   # metrics.json lookup order
STABILITY_CHANNELS: Tuple[str, ...] = ("decode_causal", "phi_log", "mu_raw")
STABILITY_K = 10                       # neighbors per word for the Jaccard overlap
_KNN_CHUNK = 2048                      # query rows per similarity block of the kNN search
_CONFIG_EXEMPT = frozenset({"seed", "data_seed"})   # config fields allowed to differ across seeds
_OPTION_EXEMPT = frozenset({"threads", "device", "drop_unknown_config_fields"})   # options that do not change results
JSON_NAME = "multiseed_semantics.json"
MARKDOWN_NAME = "MULTISEED_SEMANTICS.md"


# ----------------------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------------------

def _diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
    return None if a is None or b is None else a - b


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _fmt(value: Any, digits: int = 3) -> str:
    x = _num(value)
    return "–" if x is None else f"{x:.{digits}f}"


# ----------------------------------------------------------------------------------------------
# Runs
# ----------------------------------------------------------------------------------------------

def find_semantics_dir(run_dir: "str | Path") -> Path:
    r"""The directory holding the run's semantic ``metrics.json``, the first of :data:`SEMANTICS_SUBDIRS`."""
    run_dir = Path(run_dir)
    for sub in SEMANTICS_SUBDIRS:
        metrics = _read_json(run_dir / sub / "metrics.json")
        if metrics is not None and str(metrics.get("schema", "")).startswith("vfe3.semantics.report"):
            return run_dir / sub
    raise FileNotFoundError(f"{run_dir} has no semantic metrics.json in any of {list(SEMANTICS_SUBDIRS)}")


def discover_runs(runs: "str | Path | Sequence[str | Path]") -> List[Path]:
    r"""Run directories: the given list as is, or the children of a parent folder that hold a semantic
    ``metrics.json`` (sorted by name). Every listed run must have one (``FileNotFoundError``)."""
    if isinstance(runs, (str, Path)):
        parent = Path(runs)
        if not parent.is_dir():
            raise FileNotFoundError(f"run folder {parent} does not exist")
        found = []
        for child in sorted(p for p in parent.iterdir() if p.is_dir()):
            try:
                find_semantics_dir(child)
            except FileNotFoundError:
                continue
            found.append(child)
        return found
    out = [Path(r) for r in runs]
    for run_dir in out:
        find_semantics_dir(run_dir)
    return out


@dataclass
class SeedRecord:
    r"""One analyzed seed: its opened run, its semantic metrics and its training provenance."""

    run_dir:       Path
    semantics_dir: Path
    metrics:       Dict[str, Any] = field(repr=False)
    run:           Any = field(repr=False)                 # SemanticRun
    seed:          Optional[int]
    data_seed:     Optional[int]
    provenance:    Optional[Dict[str, Any]] = field(repr=False, default=None)
    label:         str = ""
    notes:         List[str] = field(default_factory=list)


def load_seed(run_dir: "str | Path", *, checkpoint: Optional[str] = None, vocab: Any = None,
              drop_unknown_config_fields: Optional[bool] = None) -> SeedRecord:
    r"""Open one analyzed run: ``metrics.json``, ``provenance.json`` (optional) and the checkpoint.

    The checkpoint is ``checkpoint`` or the one the analysis recorded (``metrics["run"]["checkpoint"]``,
    by file name, so a moved run still resolves), else ``best_model.pt``. The seed is the checkpoint
    config's ``seed``; the data seed is ``provenance.json["data_seed"]`` (``None`` when unrecorded).
    ``drop_unknown_config_fields`` is passed to :func:`~vfe3.semantics.run.load_semantic_run`
    (retired config toggles are dropped and noted); ``None`` takes the value the analysis recorded
    in its options (``False`` when unrecorded).
    """
    from vfe3.semantics.run import load_semantic_run

    run_dir = Path(run_dir)
    sem_dir = find_semantics_dir(run_dir)
    metrics = _read_json(sem_dir / "metrics.json") or {}
    recorded = _get(metrics, "run", "checkpoint")
    name = checkpoint or (Path(str(recorded).replace("\\", "/")).name if recorded else "best_model.pt")
    if drop_unknown_config_fields is None:
        drop_unknown_config_fields = bool(_get(metrics, "options", "drop_unknown_config_fields") or False)
    run = load_semantic_run(run_dir, checkpoint=name, vocab=vocab,
                            drop_unknown_config_fields=bool(drop_unknown_config_fields))
    provenance = _read_json(run_dir / "provenance.json")
    notes: List[str] = [f"{run_dir.name}: {n}" for n in run.notes if "no longer declares" in n]
    seed = int(run.cfg.seed)
    data_seed = None
    if provenance is not None:
        raw = provenance.get("data_seed")
        data_seed = int(raw) if isinstance(raw, int) and not isinstance(raw, bool) else None
        prov_seed = provenance.get("seed")
        if isinstance(prov_seed, int) and prov_seed != seed:
            notes.append(f"{run_dir.name}: provenance seed {prov_seed} differs from the checkpoint config seed {seed}")
    return SeedRecord(run_dir=run_dir, semantics_dir=sem_dir, metrics=metrics, run=run, seed=seed,
                      data_seed=data_seed, provenance=provenance, notes=notes)


def _labels(records: Sequence[SeedRecord]) -> List[str]:
    seeds = [r.seed for r in records]
    if all(s is not None for s in seeds) and len(set(seeds)) == len(seeds):
        return [f"seed {s}" for s in seeds]
    return [r.run_dir.name for r in records]


# ----------------------------------------------------------------------------------------------
# Homogeneity and data order
# ----------------------------------------------------------------------------------------------

def config_homogeneity(cfgs: Sequence[Any], labels: Sequence[str]) -> Dict[str, Any]:
    r"""Fields of the (dataclass) configs that differ across seeds, ``seed`` exempt.

    Returns ``{"homogeneous", "differences": {field: {label: value}}, "exempt": [...]}``.
    """
    dicts = [dataclasses.asdict(c) if dataclasses.is_dataclass(c) else dict(c) for c in cfgs]
    keys = sorted(set().union(*dicts)) if dicts else []
    differences: Dict[str, Dict[str, Any]] = {}
    for key in keys:
        if key in _CONFIG_EXEMPT:
            continue
        values = [d.get(key, "<absent>") for d in dicts]
        if any(json.dumps(v, sort_keys=True, default=repr) != json.dumps(values[0], sort_keys=True, default=repr)
               for v in values[1:]):
            differences[key] = {label: v for label, v in zip(labels, values)}
    return {"homogeneous": not differences, "differences": differences, "exempt": sorted(_CONFIG_EXEMPT)}


def option_differences(options: Sequence[Mapping[str, Any]], labels: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    r"""Semantic-analysis options (``metrics["options"]``) that differ across seeds, apart from
    :data:`_OPTION_EXEMPT` (threads and device do not change the numbers)."""
    keys = sorted(set().union(*[set(o) for o in options])) if options else []
    out: Dict[str, Dict[str, Any]] = {}
    for key in keys:
        if key in _OPTION_EXEMPT:
            continue
        values = [o.get(key, "<absent>") for o in options]
        if any(json.dumps(v, sort_keys=True, default=repr) != json.dumps(values[0], sort_keys=True, default=repr)
               for v in values[1:]):
            out[key] = {label: v for label, v in zip(labels, values)}
    return out


def data_order_status(data_seeds: Sequence[Optional[int]], seeds: Sequence[Optional[int]],
                      labels: Sequence[str]) -> Dict[str, Any]:
    r"""Whether the seeds share the training data order, with the caveat every output carries.

    ``status`` is ``"shared"`` (every run records the same effective data seed), ``"distinct"``
    (every run has its own), ``"partially_shared"`` or ``"unknown"`` (some run records none).
    """
    per = {label: ds for label, ds in zip(labels, data_seeds)}
    if any(ds is None for ds in data_seeds):
        missing = [label for label, ds in per.items() if ds is None]
        status = "unknown"
        caveat = ("The training data seed is not recorded for " + ", ".join(missing) + " (no data_seed in "
                  "provenance.json), so whether the seeds share one data order is unknown. If they do (a fixed "
                  "train_vfe3.DATA_SEED), the across-seed SD measures initialization and optimization only and is a "
                  "lower bound on the variance over data orders.")
    elif len(set(data_seeds)) == 1:
        status = "shared"
        caveat = (f"All {len(data_seeds)} seeds share one training data order (data seed {data_seeds[0]}): the "
                  "across-seed SD measures initialization and optimization only and is a lower bound on the "
                  "variance over data orders.")
    elif len(set(data_seeds)) == len(data_seeds):
        status = "distinct"
        caveat = ("Every seed has its own training data order (data seeds "
                  + ", ".join(f"{label}: {ds}" for label, ds in per.items())
                  + "), so the across-seed SD includes data-order variance.")
    else:
        status = "partially_shared"
        caveat = ("Some seeds share a training data order (data seeds "
                  + ", ".join(f"{label}: {ds}" for label, ds in per.items())
                  + "), so the across-seed SD mixes seeds with and without data-order variance.")
    equals_seed = [ds is not None and s is not None and ds == s for ds, s in zip(data_seeds, seeds)]
    return {"status": status, "data_seeds": per, "data_seed_equals_seed": all(equals_seed),
            "caveat": caveat}


# ----------------------------------------------------------------------------------------------
# Headline metrics
# ----------------------------------------------------------------------------------------------

Extractor = Callable[[Mapping[str, Any]], Optional[float]]


def _acc(battery: Mapping[str, Any], method: str, group: str = "overall") -> Optional[float]:
    return _num(_get(battery, "analogy_benchmark", group, "methods", method, "acc", "mean"))


def _macro_acc(battery: Mapping[str, Any], method: str, group: str = "overall") -> Optional[float]:
    r"""Macro-averaged (equal weight per Google category) accuracy of ``method`` when the battery
    records it (``analogy_benchmark.<group>.macro``), else ``None``."""
    return _num(_get(battery, "analogy_benchmark", group, "macro", "methods", method, "acc", "mean"))


VIEW_METRICS: Tuple[Tuple[str, str, Extractor], ...] = (
    ("google_3cosadd",       "Google 3CosAdd (micro)",           lambda b: _acc(b, "3cosadd")),
    ("google_margin",        "3CosAdd − ONLY-B (micro)",         lambda b: _diff(_acc(b, "3cosadd"), _acc(b, "only_b"))),
    ("google_3cosadd_macro", "Google 3CosAdd (macro)",           lambda b: _macro_acc(b, "3cosadd")),
    ("google_margin_macro",  "3CosAdd − ONLY-B (macro)",         lambda b: _diff(_macro_acc(b, "3cosadd"),
                                                                                 _macro_acc(b, "only_b"))),
    ("field_purity",         "field kNN purity@10",              lambda b: _num(_get(b, "fields", "purity", "purity"))),
    ("field_purity_null",    "field purity null mean",           lambda b: _num(_get(b, "fields", "purity", "null_mean"))),
    ("field_purity_z",       "field purity z",                   lambda b: _num(_get(b, "fields", "purity", "z"))),
    ("simlex999_rho",        "SimLex-999 ρ",                     lambda b: _num(_get(b, "similarity", "simlex999", "rho"))),
    ("wordsim353_rho",       "WordSim-353 ρ",                    lambda b: _num(_get(b, "similarity", "wordsim353", "rho"))),
)


def _sigma_scorer(m: Mapping[str, Any], *keys: str) -> Optional[float]:
    return _num(_get(m, "sections", "sigma", "metrics", "banks", "prior", "similarity", "simlex999", "scorers",
                     "neg_jeffreys", *keys))


def _phi(m: Mapping[str, Any], *keys: Any) -> Any:
    return _get(m, "sections", "phi", "metrics", *keys)


def _ctx_metric(m: Mapping[str, Any], *keys: Any) -> Any:
    return _get(m, "sections", "contextual", "metrics", *keys)


def _distilled(m: Mapping[str, Any], *keys: Any) -> Optional[float]:
    return _num(_ctx_metric(m, "layerwise_semantics", "layers", -1, *keys))


SECTION_METRICS: Tuple[Tuple[str, str, str, Extractor], ...] = (
    ("sigma", "sigma/cv_trace", "CV(tr Σ)",
     lambda m: _num(_get(m, "sections", "sigma", "metrics", "banks", "prior", "dispersion", "cv_trace"))),
    ("sigma", "sigma/simlex999_neg_jeffreys_rho", "SimLex-999 ρ, −Jeffreys", lambda m: _sigma_scorer(m, "rho")),
    ("sigma", "sigma/simlex999_delta_rho_jeffreys", "Δρ(−Jeffreys − (−‖Δμ‖))",
     lambda m: _sigma_scorer(m, "delta_vs_euclid_mu", "delta")),
    ("phi", "phi/group_analogy_chart_acc", "group analogy acc (chart)",
     lambda m: _num(_phi(m, "group_analogies", "benchmark", "overall", "chart", "acc"))),
    ("phi", "phi/group_analogy_only_b", "group analogy ONLY-B",
     lambda m: _num(_phi(m, "group_analogies", "benchmark", "overall", "only_b", "acc"))),
    ("phi", "phi/group_analogy_margin", "group analogy chart − ONLY-B",
     lambda m: _diff(_num(_phi(m, "group_analogies", "benchmark", "overall", "chart", "acc")),
                     _num(_phi(m, "group_analogies", "benchmark", "overall", "only_b", "acc")))),
    ("phi", "phi/positional_r2_offset", "positional R² (offset)", lambda m: _num(_phi(m, "positional", "r2_offset"))),
    ("phi", "phi/positional_r2_offset_null", "positional R² null mean",
     lambda m: _num(_phi(m, "positional", "null", "offset", "mean"))),
    ("contextual", "contextual/lens_ce_final", "lens CE, final layer (nats)",
     lambda m: _num(_ctx_metric(m, "logit_lens", "ce", -1, "mean"))),
    ("contextual", "contextual/lens_ce_unigram", "unigram readout CE (nats)",
     lambda m: _num(_ctx_metric(m, "logit_lens", "unigram_ce", "mean"))),
    ("contextual", "contextual/distilled_purity_final", "distilled purity, final layer",
     lambda m: _distilled(m, "fields", "purity")),
    ("contextual", "contextual/distilled_3cosadd_final", "distilled 3CosAdd, final layer",
     lambda m: _distilled(m, "analogies", "overall", "3cosadd", "mean")),
    ("contextual", "contextual/distilled_only_b_final", "distilled ONLY-B, final layer",
     lambda m: _distilled(m, "analogies", "overall", "only_b", "mean")),
    ("contextual", "contextual/distilled_margin_final", "distilled 3CosAdd − ONLY-B",
     lambda m: _diff(_distilled(m, "analogies", "overall", "3cosadd", "mean"),
                     _distilled(m, "analogies", "overall", "only_b", "mean"))),
)


def summarize(values: Sequence[Optional[float]]) -> Dict[str, Any]:
    r"""``{n, mean, sd, min, max}`` over the finite values; ``sd`` is the unbiased
    :math:`\sqrt{\sum_s (x_s - \bar x)^2/(n-1)}` (``None`` for :math:`n < 2`)."""
    x = np.array([v for v in (_num(v) for v in values) if v is not None], dtype=np.float64)
    n = int(x.size)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "min": None, "max": None}
    return {"n": n, "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if n >= 2 else None,
            "min": float(x.min()), "max": float(x.max())}


def common_views(metrics: Sequence[Mapping[str, Any]]) -> List[str]:
    r"""Embedding views with a battery in every seed, in :data:`report.VIEW_NAMES` order."""
    from vfe3.semantics.report import VIEW_NAMES

    sets = [set(_get(m, "sections", "embeddings", "metrics", "batteries") or {}) for m in metrics]
    shared = set.intersection(*sets) if sets else set()
    return [v for v in VIEW_NAMES if v in shared] + sorted(shared - set(VIEW_NAMES))


def headline_metrics(metrics: Sequence[Mapping[str, Any]], labels: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    r"""Every headline metric per seed and its across-seed :func:`summarize`, keyed
    ``view/<view>/<metric>`` or ``<section>/<metric>``. A metric no seed reports is left out; one some
    seeds lack lists them under ``missing``."""
    out: Dict[str, Dict[str, Any]] = {}

    def add(key: str, group: str, label: str, values: List[Optional[float]], **extra: Any) -> None:
        if all(v is None for v in values):
            return
        out[key] = {"group": group, "label": label, **extra,
                    "values": {lab: v for lab, v in zip(labels, values)},
                    "missing": [lab for lab, v in zip(labels, values) if v is None],
                    "summary": summarize(values)}

    for view in common_views(metrics):
        batteries = [_get(m, "sections", "embeddings", "metrics", "batteries", view) or {} for m in metrics]
        view_label = _get(metrics[0], "sections", "embeddings", "metrics", "views", view, "label") or view
        for name, label, extract in VIEW_METRICS:
            values = []
            for b in batteries:
                try:
                    values.append(extract(b))
                except (TypeError, KeyError):
                    values.append(None)
            add(f"view/{view}/{name}", "view", f"{view_label}: {label}", values, view=view, view_label=view_label,
                metric=name)
    for group, key, label, extract in SECTION_METRICS:
        values = []
        for m in metrics:
            try:
                values.append(extract(m))
            except (TypeError, KeyError):
                values.append(None)
        add(key, group, label, values, metric=key.split("/", 1)[1])
    return out


# ----------------------------------------------------------------------------------------------
# Stability
# ----------------------------------------------------------------------------------------------

def candidate_ids(record: SeedRecord) -> Tuple[np.ndarray, str]:
    r"""The seed's analysis candidate ids and their source.

    Read from ``<semantics dir>/candidate_ids.npy`` when the analysis saved it; otherwise rebuilt by
    :func:`~vfe3.semantics.run.prepare_context` from the options the analysis recorded (a temporary
    output directory; the same frequency floor, cut and BPE prefix filter, so the same ids)."""
    from vfe3.semantics.report import CANDIDATE_IDS_FILE, DEFAULT_OPTIONS
    from vfe3.semantics.run import prepare_context

    path = record.semantics_dir / CANDIDATE_IDS_FILE
    if path.is_file():
        return np.load(path).astype(np.int64).reshape(-1), f"{path.name} of the analysis"
    options = {**DEFAULT_OPTIONS, **dict(record.metrics.get("options") or {})}
    with tempfile.TemporaryDirectory(prefix="vfe3_multiseed_") as tmp:
        ctx = prepare_context(record.run, Path(tmp), options=options)
    return np.asarray(ctx.candidate_ids, dtype=np.int64), "rebuilt with prepare_context from the recorded options"


def shared_ids(id_lists: Sequence[np.ndarray]) -> np.ndarray:
    r"""Ids present in every list, in the order of the first."""
    common = set(int(i) for i in id_lists[0])
    for ids in id_lists[1:]:
        common &= set(int(i) for i in ids)
    return np.array([int(i) for i in id_lists[0] if int(i) in common], dtype=np.int64)


class _ChannelSkipped(Exception):
    r"""A channel does not apply to a run (the reason is the message)."""


def channel_rows(channel: str, run: Any, ids: np.ndarray, *, frame_ref_count: int = 4096
                 ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    r"""``(ids_kept, X, info)`` of one stability channel for one seed, rows in ``ids`` order.

    ``decode_causal``: :func:`report._causal_rows` of the linear decode rows ``W`` or of the Gaussian
    natural parameters (whitened over ``ids``); ``phi_log``: the flattened canonical logs of
    :func:`frames.canonical_frames` with the ``frame_ref_count`` first ids as reference set, rows
    without a real logarithm dropped; ``mu_raw``: the prior means. Raises :class:`_ChannelSkipped`
    when the channel does not apply.
    """
    from vfe3.semantics import frames
    from vfe3.semantics.report import _causal_rows, _natural_parameters

    ids = np.asarray(ids, dtype=np.int64)
    if channel == "decode_causal":
        if run.decode_kind == "linear":
            Z, rank = _causal_rows(run.decode_W(ids))
            return ids, Z, {"view": "decode_W_causal", "rank": int(rank)}
        if run.decode_kind == "gaussian":
            Z, rank = _causal_rows(_natural_parameters(run, ids))
            return ids, Z, {"view": "decode_natural_causal", "rank": int(rank)}
        raise _ChannelSkipped(f"decode kind {run.decode_kind!r} has no causal decode view")
    if channel == "phi_log":
        if not run.frames_active:
            raise _ChannelSkipped("gauge frames are inactive")
        ref = ids[:int(min(frame_ref_count, ids.size))]
        cf = frames.canonical_frames(run, ids, ref, run.log_var(ref))
        ids_ok, L = cf.log_chart()
        return ids_ok, L, {"view": "phi_log", "n_without_real_log": int(ids.size - ids_ok.size),
                           "barycenter_converged_all": cf.diagnostics.get("barycenter_converged_all"),
                           "barycenter_residual_max": cf.diagnostics.get("barycenter_residual_max")}
    if channel == "mu_raw":
        return ids, run.mu(ids), {"view": "mu_raw"}
    raise ValueError(f"unknown stability channel {channel!r}; known: {list(STABILITY_CHANNELS)}")


def knn_indices(X: np.ndarray, k: int = STABILITY_K, *, device: str = "cpu") -> np.ndarray:
    r"""(n, k) indices of the ``k`` nearest other rows by the cosine of the column-centered rows.

    Similarities are computed in blocks of :data:`_KNN_CHUNK` query rows on ``device`` (float64 on
    the CPU, float32 on a GPU); a row equal to the column mean has no direction and scores 0
    against every row.
    """
    dev = torch.device(device)
    dtype = torch.float64 if dev.type == "cpu" else torch.float32
    Z = torch.as_tensor(np.asarray(X), dtype=dtype, device=dev)
    n = int(Z.shape[0])
    if not 1 <= k <= n - 1:
        raise ValueError(f"k={k} must lie in [1, n-1] for n={n}")
    Z = Z - Z.mean(dim=0, keepdim=True)
    Z = Z / Z.norm(dim=1, keepdim=True).clamp_min(torch.finfo(dtype).tiny)   # (n, D) unit rows
    out = np.empty((n, k), dtype=np.int64)
    for start in range(0, n, _KNN_CHUNK):
        stop = min(n, start + _KNN_CHUNK)
        S = Z[start:stop] @ Z.T                                          # (c, n) cosines
        rows = torch.arange(stop - start, device=dev)
        S[rows, rows + start] = -math.inf                                # self excluded
        out[start:stop] = torch.topk(S, k, dim=1).indices.cpu().numpy()
    return out


def knn_jaccard(idx_a: np.ndarray, idx_b: np.ndarray) -> np.ndarray:
    r"""Per row :math:`|A \cap B| / |A \cup B|` of two (n, k) neighbor-index tables (sets of size k)."""
    inter = (idx_a[:, :, None] == idx_b[:, None, :]).any(axis=2).sum(axis=1)   # (n,)
    k = idx_a.shape[1]
    return inter / (2.0 * k - inter)


def jaccard_chance(n: int, k: int = STABILITY_K) -> float:
    r"""Expected Jaccard index of two independent uniform ``k``-subsets of the ``n - 1`` other words:
    :math:`\sum_i P(I = i)\, i/(2k - i)` with :math:`I` hypergeometric (population ``n - 1``, ``k``
    marked, ``k`` drawn)."""
    from scipy.stats import hypergeom

    i = np.arange(0, k + 1)
    p = hypergeom(int(n) - 1, int(k), int(k)).pmf(i)
    return float(np.sum(p * i / (2.0 * k - i)))


def _cka_self_terms(X: torch.Tensor) -> Dict[str, Any]:
    r"""Per-seed CKA terms of centered rows (computed in float64): :math:`\lVert X^\top X\rVert_F^2`,
    the row sums of squares and their total."""
    X = X.double()
    ssr = (X * X).sum(dim=1)                                             # (n,)
    return {"xtx": float(torch.linalg.matrix_norm(X.T @ X) ** 2), "ssr": ssr, "sq": float(ssr.sum())}


def _debiased_dot(xty: float, ssr_x: torch.Tensor, ssr_y: torch.Tensor, sq_x: float, sq_y: float, n: int) -> float:
    r""":math:`n(n-3)\,\mathrm{HSIC}_u = \lVert X^\top Y\rVert_F^2 - \frac{n}{n-2}\langle \mathrm{diag}\,K,
    \mathrm{diag}\,L\rangle + \frac{\lVert X\rVert_F^2\lVert Y\rVert_F^2}{(n-1)(n-2)}` for centered rows
    (the unbiased HSIC of Song et al. 2012, as in :func:`vfe3.semantics.stats.linear_cka`)."""
    return float(xty - n / (n - 2.0) * float(ssr_x @ ssr_y) + sq_x * sq_y / ((n - 1.0) * (n - 2.0)))


def cka_pair(Xa: torch.Tensor, Xb: torch.Tensor, ta: Mapping[str, Any], tb: Mapping[str, Any],
             perm: Optional[np.ndarray] = None) -> Tuple[float, float]:
    r"""``(biased, debiased)`` linear CKA of two centered row-matched tables (rows of ``Xb`` permuted by
    ``perm`` when given), in float64, reusing the per-seed terms of :func:`_cka_self_terms`."""
    n = int(Xa.shape[0])
    ssr_b = tb["ssr"]
    if perm is not None:
        index = torch.as_tensor(perm, device=Xb.device)
        Xb, ssr_b = Xb[index], ssr_b[index]
    xty = float(torch.linalg.matrix_norm(Xa.double().T @ Xb.double()) ** 2)       # ||X_b^T X_a||_F^2
    biased = xty / math.sqrt(ta["xtx"] * tb["xtx"])
    den_a = _debiased_dot(ta["xtx"], ta["ssr"], ta["ssr"], ta["sq"], ta["sq"], n)
    den_b = _debiased_dot(tb["xtx"], tb["ssr"], tb["ssr"], tb["sq"], tb["sq"], n)
    num = _debiased_dot(xty, ta["ssr"], ssr_b, ta["sq"], tb["sq"], n)
    debiased = num / math.sqrt(den_a * den_b) if den_a > 0.0 and den_b > 0.0 else float("nan")
    return biased, debiased


def _pair_summary(M: np.ndarray) -> Dict[str, Optional[float]]:
    S = M.shape[0]
    vals = np.array([M[a, b] for a in range(S) for b in range(a + 1, S)], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"mean": None, "min": None, "max": None, "n_pairs": 0}
    return {"mean": float(vals.mean()), "min": float(vals.min()), "max": float(vals.max()), "n_pairs": int(vals.size)}


_MU_RAW_NOTE = {
    "exact": ("μ raw: the global positive-diagonal rescaling μ → Dμ is an exact redundancy of this configuration and "
              "is not in the invariance class of kNN-cosine or CKA, so this channel's stability also reflects each "
              "seed's arbitrary diagonal scale"),
    "assumed": ("μ raw: the global positive-diagonal rescaling μ → Dμ is assumed (not verified) to be a redundancy of "
                "this configuration and is not in the invariance class of kNN-cosine or CKA, so this channel's "
                "stability may reflect each seed's arbitrary diagonal scale"),
    "broken": ("μ raw: the global positive-diagonal rescaling is not a symmetry of this configuration ({reason}), so "
               "the raw coordinates are not a gauge choice and the stability compares the trained means directly"),
}


def stability_analysis(records: Sequence[SeedRecord], labels: Sequence[str], *, device: str = "cpu",
                       channels: Sequence[str] = STABILITY_CHANNELS, k: int = STABILITY_K,
                       log: Callable[[str], None] = print) -> Dict[str, Any]:
    r"""kNN Jaccard@``k`` and linear CKA between every pair of seeds, per channel (module docstring).

    Returns ``{"shared_candidates", "candidate_sources", "channels": {channel: record}}``; a channel
    that does not apply to some seed, or fails, is recorded with its reason and the others still run.
    """
    ids_per_seed, sources = [], []
    for r in records:
        ids, source = candidate_ids(r)
        ids_per_seed.append(ids)
        sources.append(source)
    shared = shared_ids(ids_per_seed)
    out: Dict[str, Any] = {"k": int(k), "device": str(device), "n_candidates": [int(i.size) for i in ids_per_seed],
                           "n_shared_candidates": int(shared.size),
                           "candidate_sources": dict(zip(labels, sources)), "channels": {}}
    frame_ref = int(_get(records[0].metrics, "options", "frame_ref_count") or 4096)
    d_status, d_reason = diagonal_redundancy_status(records[0].run.cfg)
    dev = torch.device(device)
    for channel in channels:
        t0 = time.perf_counter()
        rec: Dict[str, Any] = {"status": "ok"}
        try:
            rows: List[Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]] = []
            for r in records:                                           # tables kept in float32 (memory)
                ids_kept, X, info = channel_rows(channel, r.run, shared, frame_ref_count=frame_ref)
                rows.append((ids_kept, np.asarray(X, dtype=np.float32), info))
                del X
            common = shared_ids([ids for ids, _, _ in rows])
            if common.size < k + 2:
                raise _ChannelSkipped(f"only {common.size} words are shared by every seed")
            S, n = len(rows), int(common.size)
            dims: List[int] = []
            neighbors: List[np.ndarray] = []
            centered: List[torch.Tensor] = []                         # (n, D_s) float32, one per seed
            terms: List[Dict[str, Any]] = []
            for s in range(S):                                          # one seed's table in memory at a time
                ids, X, _ = rows[s]
                position = {int(t): i for i, t in enumerate(ids)}
                X = np.asarray(X)[[position[int(t)] for t in common]]  # (n, D_s)
                rows[s] = (ids, None, rows[s][2])                       # release the full table
                dims.append(int(X.shape[1]))
                neighbors.append(knn_indices(X, k, device=device))
                Xc = torch.as_tensor(X, dtype=torch.float64, device=dev)
                Xc = (Xc - Xc.mean(dim=0, keepdim=True)).float()      # centered in float64, stored in float32
                terms.append(_cka_self_terms(Xc))                       # from the stored values (consistent)
                centered.append(Xc)
                del X, Xc
            jac, jac_median = np.eye(S), np.eye(S)
            names = ("cka_biased", "cka_debiased", "cka_biased_row_permuted", "cka_debiased_row_permuted")
            mats = {name: np.full((S, S), np.nan) for name in names}
            for a in range(S):
                mats["cka_biased"][a, a] = mats["cka_debiased"][a, a] = 1.0
                for b in range(a + 1, S):
                    per_word = knn_jaccard(neighbors[a], neighbors[b])
                    jac[a, b] = jac[b, a] = float(per_word.mean())
                    jac_median[a, b] = jac_median[b, a] = float(np.median(per_word))
                    biased, debiased = cka_pair(centered[a], centered[b], terms[a], terms[b])
                    perm = np.random.default_rng([0, a, b]).permutation(n)
                    pb, pd = cka_pair(centered[a], centered[b], terms[a], terms[b], perm=perm)
                    for name, value in zip(names, (biased, debiased, pb, pd)):
                        mats[name][a, b] = mats[name][b, a] = value
            rec.update({
                "view": rows[0][2].get("view"), "n_rows": n, "dims": dims,
                "labels": list(labels), "jaccard": jac, "jaccard_median": jac_median,
                "jaccard_chance": jaccard_chance(n, k), **mats,
                "pairs": {"jaccard": _pair_summary(jac), **{name: _pair_summary(M) for name, M in mats.items()}},
                "per_seed": {label: info for label, (_, _, info) in zip(labels, rows)},
            })
            if channel == "mu_raw":
                rec["gauge_note"] = _MU_RAW_NOTE[d_status].format(reason=d_reason)
            elif channel == "phi_log":
                rec["gauge_note"] = ("φ log chart: invariant under the split redundancy and the diagonal rescaling "
                                     "(at the barycenter fixed point), so each seed's rows do not depend on its "
                                     "trivialization")
                if not all(bool(info.get("barycenter_converged_all", True)) for _, _, info in rows):
                    rec["gauge_note"] += ("; the barycenter did not converge in every block of some seed, so this "
                                          "holds there only up to the barycenter residual")
            else:
                rec["gauge_note"] = ("causal decode rows: every global invertible reparameterization of the belief "
                                     "coordinates acts on them as an orthogonal map, so both metrics are exactly "
                                     "invariant under each seed's trivialization")
        except _ChannelSkipped as exc:
            rec = {"status": "skipped", "reason": str(exc)}
        except Exception as exc:                        # recorded; the other channels still run
            rec = {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc()[-3000:]}
        rec["seconds"] = time.perf_counter() - t0
        out["channels"][channel] = rec
        log(f"[multiseed] stability {channel}: {rec['status']} ({rec['seconds']:.1f} s)")
    return out


# ----------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------

def _rows_per_view(entries: Sequence[Mapping[str, Any]]) -> Tuple[List[Optional[Mapping[str, Any]]], int]:
    r"""The view entries padded with ``None`` so that every view starts a new row of the grid, and the
    column count (the metrics of one view on one row when at most five, else on two rows)."""
    groups: Dict[str, List[Mapping[str, Any]]] = {}
    for entry in entries:
        groups.setdefault(entry["view"], []).append(entry)
    per_view = max((len(g) for g in groups.values()), default=1)
    ncols = per_view if per_view <= 5 else math.ceil(per_view / 2)
    padded: List[Optional[Mapping[str, Any]]] = []
    for group in groups.values():
        padded += list(group) + [None] * (-len(group) % ncols)
    return padded, ncols


def _metric_grid(entries: Sequence[Optional[Mapping[str, Any]]], labels: Sequence[str], out: Path, name: str,
                 title: str, ncols: int) -> List[Path]:
    r"""One small axis per metric: each seed's value (dot, x = seed), the across-seed mean (line)
    and :math:`\pm 1` SD (band). A ``None`` entry leaves its grid cell empty."""
    import textwrap

    import matplotlib.pyplot as plt

    ncols = max(1, min(int(ncols), len(entries)))
    nrows = math.ceil(len(entries) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2, 1.25 * nrows + 0.7), squeeze=False,
                             constrained_layout=True)
    x = np.arange(len(labels))
    for ax, entry in zip(axes.ravel(), entries):
        if entry is None:
            ax.set_visible(False)
            continue
        values = np.array([np.nan if entry["values"][lab] is None else entry["values"][lab] for lab in labels],
                          dtype=np.float64)
        summ = entry["summary"]
        if summ["mean"] is not None:
            if summ["sd"] is not None:
                ax.axhspan(summ["mean"] - summ["sd"], summ["mean"] + summ["sd"], color=OKABE_ITO[0], alpha=0.15, lw=0)
            ax.axhline(summ["mean"], color=OKABE_ITO[0], lw=1.0)
        ax.plot(x, values, "o", color=OKABE_ITO[1], ms=3.5)
        ax.set_xticks(x)
        ax.set_xticklabels([lab.replace("seed ", "s") for lab in labels], fontsize=6)
        ax.set_xlim(-0.6, len(labels) - 0.4)
        ax.tick_params(axis="y", labelsize=6)
        sd = "–" if summ["sd"] is None else f"{summ['sd']:.3g}"
        ax.set_title(f"{entry['label']}\nmean {_fmt(summ['mean'])}, SD {sd}", fontsize=6)
    for ax in axes.ravel()[len(entries):]:
        ax.set_visible(False)
    fig.suptitle(textwrap.fill(title, 150), fontsize=7)
    return save_figure(fig, SimpleNamespace(figures_dir=out), "figures", name)


def _plot_stability(stab: Mapping[str, Any], labels: Sequence[str], out: Path) -> List[Path]:
    import matplotlib.pyplot as plt

    ok = [(c, r) for c, r in stab["channels"].items() if r.get("status") == "ok"]
    if not ok:
        return []
    S = len(labels)
    fig, axes = plt.subplots(2, len(ok), figsize=(min(7.2, 2.4 * len(ok) + 0.6), 4.6), squeeze=False,
                             constrained_layout=True)
    ticks = [lab.replace("seed ", "s") for lab in labels]
    for j, (channel, rec) in enumerate(ok):
        for i, (key, what, ref) in enumerate((
                ("jaccard", f"kNN Jaccard@{stab['k']}", f"chance {rec['jaccard_chance']:.4f}"),
                ("cka_debiased", "debiased linear CKA",
                 f"row-permuted {_fmt(rec['pairs']['cka_debiased_row_permuted']['mean'])}"))):
            ax = axes[i, j]
            M = np.asarray(rec[key], dtype=np.float64)
            im = ax.imshow(M, vmin=0.0, vmax=1.0, cmap="viridis")
            for a in range(S):
                for b in range(S):
                    if np.isfinite(M[a, b]):
                        ax.text(b, a, f"{M[a, b]:.2f}", ha="center", va="center", fontsize=6,
                                color="white" if M[a, b] < 0.6 else "black")
            ax.set_xticks(range(S))
            ax.set_yticks(range(S))
            ax.set_xticklabels(ticks, fontsize=6)
            ax.set_yticklabels(ticks, fontsize=6)
            ax.set_title(f"{rec.get('view') or channel}: {what}\n(n = {rec['n_rows']}; {ref})", fontsize=6)
        fig.colorbar(im, ax=axes[:, j], shrink=0.6)
    fig.suptitle("Cross-seed stability (invariant under each seed's gauge trivialization; see the report)", fontsize=7)
    return save_figure(fig, SimpleNamespace(figures_dir=out), "figures", "stability")


# ----------------------------------------------------------------------------------------------
# Markdown
# ----------------------------------------------------------------------------------------------

def _render_markdown(payload: Mapping[str, Any], labels: Sequence[str], out: Path) -> str:
    lines: List[str] = []
    add = lines.append
    add("# Multi-seed semantic geometry\n")
    add(f"{len(labels)} seeds, created {payload['created']}.\n")
    add(f"**Data order.** {payload['data_order']['caveat']}\n")
    hom = payload["config_homogeneity"]
    if hom["homogeneous"]:
        add("**Configuration.** The checkpoint configurations agree in every field except the seed.\n")
    else:
        add("**Configuration differs across runs** (fields other than the seed): "
            + "; ".join(f"`{k}`: " + ", ".join(f"{lab} {v!r}" for lab, v in vals.items())
                        for k, vals in hom["differences"].items())
            + ". The across-seed statistics below mix these configurations.\n")
    if payload["analysis_option_differences"]:
        add("**Analysis options differ across runs:** "
            + "; ".join(f"`{k}`: " + ", ".join(f"{lab} {v!r}" for lab, v in vals.items())
                        for k, vals in payload["analysis_option_differences"].items()) + ".\n")

    add("## Runs\n")
    add(markdown_table([{"seed": lab, "run": Path(r["run_dir"]).name, "data seed": r["data_seed"],
                         "preset": r["preset"], "checkpoint": r["checkpoint"],
                         "semantics": Path(r["semantics_dir"]).name}
                        for lab, r in zip(labels, payload["runs"])],
                       ["seed", "run", "data seed", "preset", "checkpoint", "semantics"]))

    add("## Headline metrics across seeds\n")
    add("Mean, unbiased SD (n − 1), minimum and maximum over the seeds that report the metric; view "
        "metrics are the full-vocabulary battery of each view, contextual distilled metrics retrieve among "
        "the collected words only.\n")
    rows = []
    for key, entry in payload["metrics"].items():
        s = entry["summary"]
        row = {"metric": entry["label"], **{lab: entry["values"].get(lab) for lab in labels},
               "mean": s["mean"], "SD": s["sd"], "min": s["min"], "max": s["max"], "n": s["n"]}
        rows.append(row)
    add(markdown_table(rows, ["metric", *labels, "mean", "SD", "min", "max", "n"]) if rows else "_No metric found._\n")

    add("## Cross-seed stability\n")
    stab = payload.get("stability") or {}
    if stab.get("status") in ("skipped", "failed"):
        add(f"Stability was {stab['status']}: {stab.get('reason') or stab.get('error')}.\n")
    else:
        add(f"On the {stab.get('n_shared_candidates')} candidate words every seed shares ("
            + ", ".join(f"{lab}: {src}" for lab, src in (stab.get("candidate_sources") or {}).items())
            + f"), kNN Jaccard@{stab.get('k')} of the cosine neighbors of the centered rows and linear CKA between "
            "every pair of seeds. Both are invariant under a separate translation, orthogonal map and positive "
            "scale of each seed's rows, and the channels are chosen so that each seed's gauge trivialization acts "
            "inside that class (module docstring of vfe3.semantics.multiseed).\n")
        srows = []
        for channel, rec in (stab.get("channels") or {}).items():
            if rec.get("status") != "ok":
                srows.append({"channel": channel, "status": f"{rec.get('status')}: "
                              + str(rec.get("reason") or rec.get("error"))})
                continue
            p = rec["pairs"]

            def rng(entry: Mapping[str, Any]) -> str:
                return f"{_fmt(entry['mean'])} [{_fmt(entry['min'])}, {_fmt(entry['max'])}]"

            srows.append({"channel": channel, "status": "ok", "view": rec.get("view"), "words": rec["n_rows"],
                          "D per seed": "/".join(str(d) for d in rec["dims"]),
                          "Jaccard mean [min, max]": rng(p["jaccard"]), "Jaccard chance": f"{rec['jaccard_chance']:.4f}",
                          "debiased CKA [min, max]": rng(p["cka_debiased"]),
                          "row-permuted debiased": _fmt(p["cka_debiased_row_permuted"]["mean"]),
                          "biased CKA": _fmt(p["cka_biased"]["mean"]),
                          "row-permuted biased": _fmt(p["cka_biased_row_permuted"]["mean"])})
        cols = ["channel", "status", "view", "words", "D per seed", "Jaccard mean [min, max]", "Jaccard chance",
                "debiased CKA [min, max]", "row-permuted debiased", "biased CKA", "row-permuted biased"]
        add(markdown_table(srows, cols) if srows else "_No channel ran._\n")
        for channel, rec in (stab.get("channels") or {}).items():
            if rec.get("gauge_note"):
                add(f"- {channel}: {rec['gauge_note']}.")
        add("")
        add("Neighbor tables of a single seed are illustrations, not stable findings (Antoniak and Mimno 2018; "
            "Wendlandt et al. 2018).\n")

    add("## Figures\n")
    if payload["figures"]:
        for rel in payload["figures"]:
            if rel.endswith(".png"):
                add(f"- [{rel}]({rel})")
        add("")
    else:
        add("_No figure was written._\n")
    if payload["notes"]:
        add("## Notes\n")
        for note in payload["notes"]:
            add(f"- {note}")
        add("")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------------------------

def _default_out_dir(runs: Any, run_dirs: Sequence[Path]) -> Path:
    if isinstance(runs, (str, Path)):
        return Path(runs) / "multiseed_semantics"
    parents = {p.resolve().parent for p in run_dirs}
    base = parents.pop() if len(parents) == 1 else Path(run_dirs[0]).resolve().parent
    return base / "multiseed_semantics"


def aggregate_seeds(
    runs:       "str | Path | Sequence[str | Path]",

    *,
    out_dir:    "str | Path | None" = None,
    checkpoint: Optional[str] = None,
    stability:  bool = True,
    device:     str = "cpu",
    threads:    Optional[int] = None,
    vocab:      Any = None,
    drop_unknown_config_fields: Optional[bool] = None,
    log:        Callable[[str], None] = print,
) -> Path:
    r"""Aggregate the semantic analyses of two or more seeds; returns the path of ``MULTISEED_SEMANTICS.md``.

    ``runs`` is a list of run directories or one parent folder whose children are the runs
    (:func:`discover_runs`). ``out_dir`` defaults to ``<parent>/multiseed_semantics``.
    ``checkpoint`` overrides the checkpoint each analysis recorded. ``stability=False`` skips the
    cross-seed stability, which rebuilds each seed's channel tables (the frame chart is the costly
    part). ``device`` runs the stability kNN and CKA (``"cpu"`` or ``"cuda[:i]"``); ``threads`` pins
    torch's intra-op threads for the call. ``vocab`` overrides the tokenizer vocabulary (tests).
    ``drop_unknown_config_fields`` reopens checkpoints whose configs carry retired toggles
    (:func:`load_seed`; ``None`` follows each analysis's recorded option); the dropped fields are
    listed in the notes, and the homogeneity check compares the configs after the drop.
    """
    started = datetime.datetime.now()
    run_dirs = discover_runs(runs)
    if len(run_dirs) < 2:
        raise ValueError(f"need at least two analyzed runs, found {len(run_dirs)} in {runs!r}")
    out = Path(out_dir) if out_dir is not None else _default_out_dir(runs, run_dirs)
    out.mkdir(parents=True, exist_ok=True)
    previous_threads = torch.get_num_threads()
    if threads is not None:
        torch.set_num_threads(int(threads))
    try:
        records = [load_seed(d, checkpoint=checkpoint, vocab=vocab,
                             drop_unknown_config_fields=drop_unknown_config_fields) for d in run_dirs]
        labels = _labels(records)
        for r, lab in zip(records, labels):
            r.label = lab
        log(f"[multiseed] {len(records)} runs: " + ", ".join(f"{lab} ({r.run_dir.name})" for lab, r in zip(labels, records)))
        notes: List[str] = [n for r in records for n in r.notes]
        if isinstance(runs, (str, Path)):
            unanalyzed = sorted(p.name for p in Path(runs).iterdir()
                                if p.is_dir() and p not in run_dirs and (p / "best_model.pt").is_file())
            if unanalyzed:
                notes.append("run folders with a checkpoint but no semantic metrics.json were left out: "
                             + ", ".join(unanalyzed))
        seeds = [r.seed for r in records]
        if len(set(seeds)) < len(seeds):
            notes.append("two or more runs share a seed, so they are not independent replicates: "
                         + ", ".join(f"{r.run_dir.name} = {r.seed}" for r in records))
        homogeneity = config_homogeneity([r.run.cfg for r in records], labels)
        opt_diff = option_differences([r.metrics.get("options") or {} for r in records], labels)
        data_order = data_order_status([r.data_seed for r in records], seeds, labels)
        table = headline_metrics([r.metrics for r in records], labels)
        for r, lab in zip(records, labels):
            failed = [name for name, sec in (r.metrics.get("sections") or {}).items()
                      if isinstance(sec, Mapping) and sec.get("status") != "ok"]
            if failed:
                notes.append(f"{lab}: sections not ok: " + ", ".join(
                    f"{name} ({r.metrics['sections'][name].get('status')})" for name in failed))
        if stability:
            try:
                stab = stability_analysis(records, labels, device=device, log=log)
            except Exception as exc:                    # recorded; the headline table survives
                stab = {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc()[-3000:]}
                notes.append(f"stability failed: {type(exc).__name__}: {exc}")
        else:
            stab = {"status": "skipped", "reason": "stability=False"}

        figures: List[str] = []
        figure_errors: Dict[str, str] = {}
        apply_style()
        view_entries = [e for e in table.values() if e["group"] == "view"]
        section_entries = [e for e in table.values() if e["group"] != "view"]
        view_cells, view_cols = _rows_per_view(view_entries)
        caveat = data_order["caveat"]
        jobs = [
            ("headline_views", lambda: _metric_grid(view_cells, labels, out, "headline_views",
                                                    f"Headline view metrics per seed (line: mean, band: ±1 SD). "
                                                    f"{caveat}", ncols=view_cols)
             if view_entries else []),
            ("headline_sections", lambda: _metric_grid(section_entries, labels, out, "headline_sections",
                                                       f"Σ, φ and contextual headline metrics per seed (line: mean, "
                                                       f"band: ±1 SD). {caveat}", ncols=4)
             if section_entries else []),
            ("stability", lambda: _plot_stability(stab, labels, out) if stab.get("channels") else []),
        ]
        for name, job in jobs:
            try:
                figures += [p.relative_to(out).as_posix() for p in job()]
            except Exception as exc:                    # recorded; the other outputs survive
                figure_errors[name] = traceback.format_exc()[-3000:]
                notes.append(f"figure {name} failed: {type(exc).__name__}: {exc}")

        payload: Dict[str, Any] = {
            "schema": "vfe3.semantics.multiseed/1",
            "created": started.isoformat(timespec="seconds"),
            "runs": [{"label": lab, "run_dir": str(r.run_dir), "semantics_dir": str(r.semantics_dir),
                      "checkpoint": r.run.checkpoint_path.name, "seed": r.seed, "data_seed": r.data_seed,
                      "preset": _get(r.metrics, "options", "preset") or "full (pre-preset analysis)",
                      "n_candidates": _get(r.metrics, "run", "n_candidates")}
                     for lab, r in zip(labels, records)],
            "config_homogeneity": homogeneity,
            "analysis_option_differences": opt_diff,
            "data_order": data_order,
            "diagonal_redundancy": dict(zip(("status", "reason"), diagonal_redundancy_status(records[0].run.cfg))),
            "metrics": table,
            "stability": stab,
            "figures": figures,
            "figure_errors": figure_errors,
            "notes": notes,
        }
        write_json(out / JSON_NAME, payload)
        md_path = out / MARKDOWN_NAME
        md_path.write_text(_render_markdown(payload, labels, out), encoding="utf-8")
        log(f"[multiseed] wrote {out / JSON_NAME} and {md_path}")
        return md_path
    finally:
        torch.set_num_threads(previous_threads)
