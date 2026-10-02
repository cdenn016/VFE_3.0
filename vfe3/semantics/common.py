r"""Shared contracts for :mod:`vfe3.semantics`.

Every analysis module exchanges data through the types defined here:

* :class:`VectorView` -- one token-indexed vector chart (a table row per token) that the standard
  embedding battery (neighbors, analogies, offsets, directions, clustering, geometry) runs on.
* :class:`AnalogyQuestion` -- one resolved analogy ``a : a_star :: b : b_star`` in token ids.
* :class:`AnalysisContext` -- run-wide inputs shared by all sections (analysis vocabulary,
  resolved lexicon/benchmarks, output directories, options, cross-section cache).
* :class:`SectionResult` -- what a section returns to the report writer.

It also owns the figure style, the atomic PNG+PDF figure writer, the markdown table writer and the
strict JSON writer, so every section emits artifacts the same way.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np


OKABE_ITO: Tuple[str, ...] = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000",
)


class AnalogyQuestion(NamedTuple):
    r"""One analogy ``a : a_star :: b : b_star`` as token ids; ``b_star`` is the answer."""

    category: str
    a:        int
    a_star:   int
    b:        int
    b_star:   int


@dataclass
class VectorView:
    r"""A token-indexed vector chart: row ``r`` of ``X`` is the vector of token ``ids[r]``.

    ``metric`` selects the similarity the battery uses: ``"cosine"`` (rows are compared by angle,
    after the battery's optional centering) or ``"euclidean"`` (rows are compared by distance; used
    for charts whose Euclidean geometry is itself meaningful, e.g. the diagonal-Gaussian W2 chart
    ``(mu, sqrt(var))`` where Euclidean distance *is* the 2-Wasserstein distance).
    ``note`` states the chart/gauge status of the view for the report.
    """

    name:   str                  # file-name slug, e.g. "mu_raw"
    label:  str                  # display label for figures and tables
    ids:    np.ndarray           # (n,) int64 token ids, unique
    X:      np.ndarray           # (n, D) float32 rows
    metric: str = "cosine"       # "cosine" | "euclidean"
    note:   str = ""             # chart / invariance statement for the report
    group:  str = ""             # channel family: "mu" | "sigma" | "phi" | "decode" | "contextual"

    def __post_init__(self) -> None:
        self.ids = np.asarray(self.ids, dtype=np.int64).reshape(-1)
        self.X = np.asarray(self.X, dtype=np.float32)
        if self.X.ndim != 2 or self.X.shape[0] != self.ids.shape[0]:
            raise ValueError(
                f"view {self.name!r}: X must be (n, D) with n == len(ids); "
                f"got X {self.X.shape}, ids {self.ids.shape}")
        if self.metric not in ("cosine", "euclidean"):
            raise ValueError(f"view {self.name!r}: unknown metric {self.metric!r}")
        if np.unique(self.ids).size != self.ids.size:
            raise ValueError(f"view {self.name!r}: token ids must be unique")
        if not np.isfinite(self.X).all():
            raise ValueError(f"view {self.name!r}: X contains non-finite values")
        self._row_of: Dict[int, int] = {int(t): r for r, t in enumerate(self.ids)}

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    @property
    def dim(self) -> int:
        return int(self.X.shape[1])

    def rows(self, token_ids: Sequence[int]) -> np.ndarray:
        r"""Row index of each token id, ``-1`` where the token is not in the view."""
        return np.array([self._row_of.get(int(t), -1) for t in token_ids], dtype=np.int64)

    def has(self, token_id: int) -> bool:
        return int(token_id) in self._row_of

    def subset(self, token_ids: Sequence[int], *, name: Optional[str] = None) -> "VectorView":
        r"""The view restricted to ``token_ids`` that it contains, in the given order."""
        rows = self.rows(token_ids)
        rows = rows[rows >= 0]
        return VectorView(name=name or self.name, label=self.label, ids=self.ids[rows],
                          X=self.X[rows], metric=self.metric, note=self.note, group=self.group)


@dataclass
class SectionResult:
    r"""What one section hands the report writer.

    ``metrics`` must be JSON-serializable after :func:`to_jsonable` (numpy scalars/arrays allowed).
    ``tables`` maps a table name to markdown text (already written under ``tables_dir``).
    ``headline`` holds the few numbers the report's summary table shows, as ``label -> value``.
    ``status`` is ``"ok"``, ``"skipped"`` (not applicable to this run; ``notes`` says why) or
    ``"failed"`` (``error`` holds the traceback text).
    """

    name:     str
    metrics:  Dict[str, Any] = field(default_factory=dict)
    tables:   Dict[str, str] = field(default_factory=dict)
    figures:  List[Path] = field(default_factory=list)
    notes:    List[str] = field(default_factory=list)
    headline: Dict[str, Any] = field(default_factory=dict)
    status:   str = "ok"
    error:    Optional[str] = None
    seconds:  Optional[float] = None


@dataclass
class AnalysisContext:
    r"""Run-wide inputs shared by every section.

    ``candidate_ids`` is the analysis vocabulary: whole-word tokens that pass the frequency floor,
    ordered by descending training count. Every resolved resource below is a subset of it.
    """

    out_dir:           Path
    figures_dir:       Path
    tables_dir:        Path
    vocab:             Any                                   # vfe3.semantics.vocab.Vocab
    candidate_ids:     np.ndarray                            # (n,) int64, frequency-descending
    counts:            Optional[np.ndarray]                  # (V,) int64 training counts or None
    lexicon:           Dict[str, List[int]]                  # semantic field -> token ids
    relations:         Dict[str, List[Tuple[int, int]]]      # relation -> (a, a_star) pairs
    analogies:         List[AnalogyQuestion]                 # benchmark (Google) questions
    curated_analogies: List[AnalogyQuestion]                 # famous analogies for example tables
    similarity:        Dict[str, List[Tuple[int, int, float]]]  # benchmark -> (w1, w2, human score)
    hypernyms:         List[Tuple[int, int]]                 # (hyponym, hypernym)
    axes:              Dict[str, Dict[str, Any]]             # semantic-projection axes (see resources)
    probe_ids:         List[int]                             # words for nearest-neighbor tables
    ambiguous_ids:     List[int]                             # polysemous words for contextual clouds
    coverage:          Dict[str, Any]                        # resolution coverage per resource
    seed:              int = 0
    device:            str = "cpu"
    threads:           int = 1
    options:           Dict[str, Any] = field(default_factory=dict)
    cache:             Dict[str, Any] = field(default_factory=dict)

    def display(self, token_id: int) -> str:
        r"""Short human-readable label of one token id (delegates to the vocabulary)."""
        return self.vocab.display(int(token_id))

    def log_count(self, token_ids: Sequence[int]) -> Optional[np.ndarray]:
        r"""``log(1 + count)`` of each token, or ``None`` when counts are unavailable."""
        if self.counts is None:
            return None
        return np.log1p(self.counts[np.asarray(token_ids, dtype=np.int64)].astype(np.float64))


# ----------------------------------------------------------------------------------------------
# Gauge status of the global positive-diagonal rescaling
# ----------------------------------------------------------------------------------------------

def diagonal_redundancy_status(cfg: Any) -> Tuple[str, str]:
    r"""Whether the global positive-diagonal rescaling is an exact redundancy of this model.

    The rescaling acts as $\mu \mapsto D\mu$, $\log\sigma^2 \mapsto \log\sigma^2 + 2\log D$,
    $e^{X_v} \mapsto D e^{X_v}$ with compensating changes of the coordinate MLP and the decoder. It
    was measured to leave the logits unchanged (to fp32 round-off) only for the probed class:
    ``gauge_group="block_glk"``, ``attention_mode="projected_predictive"``, the coordinate block MLP
    (or none), no head mixer, no CG coupling, no RoPE, and no coordinate normalization.

    Returns ``(status, reason)`` with ``status`` one of

    * ``"exact"`` -- the configuration is in the probed class;
    * ``"broken"`` -- a structural reason rules it out: a LayerNorm (affine or not) normalizes
      across coordinates, so $\mathrm{LN}(D\mu) \ne D\,\mathrm{LN}(\mu)$; or $D e^{X}$ leaves the
      gauge group (compact/symplectic groups, tied or traceless variants), so the compensated frame
      is not representable by ``phi_embed``;
    * ``"assumed"`` -- nothing rules it out but the configuration was not probed (e.g. a
      Mahalanobis norm, which is equivariant, or a head mixer / CG coupling / RoPE / other block MLP).
    """
    norms = (getattr(cfg, "norm_type_block", "none"), getattr(cfg, "norm_type_final", "none"))
    if "layernorm" in norms:
        return "broken", ("a LayerNorm (affine or not) normalizes across coordinates, so "
                          "LN(Dμ) ≠ D LN(μ) for a non-uniform D")
    group = getattr(cfg, "gauge_group", "block_glk")
    if group != "block_glk":
        return "broken", f"D e^X leaves the gauge group {group!r}, so the compensated frame is not representable"
    unprobed = []
    if getattr(cfg, "attention_mode", None) != "projected_predictive":
        unprobed.append(f"attention_mode={getattr(cfg, 'attention_mode', None)!r}")
    if getattr(cfg, "use_block_mlp", False) and getattr(cfg, "block_mlp_mode", "coordinate") != "coordinate":
        unprobed.append(f"block_mlp_mode={cfg.block_mlp_mode!r}")
    for flag in ("use_head_mixer", "use_cg_coupling", "use_priorbank_head_evidence_mixer"):
        if getattr(cfg, flag, False):
            unprobed.append(f"{flag}=True")
    if getattr(cfg, "pos_rotation", "none") != "none":
        unprobed.append(f"pos_rotation={cfg.pos_rotation!r}")
    if "mahalanobis" in norms:
        unprobed.append("a Mahalanobis norm (equivariant, not measured)")
    if unprobed:
        return "assumed", "not probed for " + ", ".join(unprobed)
    return "exact", ("measured on projected-predictive block-GL runs with a coordinate MLP and no "
                     "coordinate normalization")


# ----------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------

def _regular_weight_fonts() -> List[str]:
    r"""Sans-serif families with an upright regular (weight 400) face, Arial first.

    A family whose only upright file is bold (e.g. a system Noto Sans installed as
    ``NotoSans-Bold.ttf`` alone) would render every label bold, so such families are dropped.
    CJK/Arabic families with a regular face stay as glyph fallbacks.
    """
    from matplotlib import font_manager
    from vfe3.viz.text import installed_multilingual_sans_serif

    regular = {entry.name for entry in font_manager.fontManager.ttflist
               if entry.style == "normal" and entry.weight in (400, "normal", "regular")}
    ordered = ["Arial"] + [name for name in installed_multilingual_sans_serif() if name != "Arial"]
    fonts = [name for name in ordered if name in regular]
    return fonts or ["DejaVu Sans"]


def apply_style() -> None:
    r"""Colorblind-safe, publication-oriented Matplotlib defaults (idempotent)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fonts = _regular_weight_fonts()
    plt.rcParams.update({
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "font.size":         9,
        "font.family":       fonts,
        "font.sans-serif":   fonts,
        "axes.titlesize":    10,
        "axes.labelsize":    9,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "legend.fontsize":   8,
        "legend.frameon":    False,
        "axes.grid":         False,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.prop_cycle":   matplotlib.cycler(color=list(OKABE_ITO)),
        "image.cmap":        "viridis",
        "pdf.fonttype":      42,
    })


def save_figure(fig: Any, ctx: AnalysisContext, section: str, name: str) -> List[Path]:
    r"""Write ``fig`` as ``<figures_dir>/<section>/<name>.png`` (300 dpi) and ``.pdf``, atomically.

    Returns the written paths and closes the figure.
    """
    import matplotlib.pyplot as plt

    target_dir = Path(ctx.figures_dir) / section
    target_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    try:
        for suffix in (".png", ".pdf"):
            target = target_dir / f"{name}{suffix}"
            temporary = target.with_name(f".{target.stem}.tmp{suffix}")
            fig.savefig(str(temporary))
            os.replace(temporary, target)
            written.append(target)
    finally:
        plt.close(fig)
    return written


def panel_label(ax: Any, letter: str) -> None:
    r"""Bold panel letter at the top-left corner of ``ax``."""
    ax.text(-0.12, 1.06, letter, transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="bottom", ha="left")


# ----------------------------------------------------------------------------------------------
# Tables and JSON
# ----------------------------------------------------------------------------------------------

def format_value(value: Any, digits: int = 3) -> str:
    r"""Compact cell text: floats to ``digits`` significant decimals, ``None``/NaN to an en dash."""
    if value is None:
        return "–"
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return "–"
        return f"{float(value):.{digits}f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown_table(rows: Sequence[Mapping[str, Any]], columns: Optional[Sequence[str]] = None,
                   *, digits: int = 3) -> str:
    r"""Render ``rows`` (dicts) as a GitHub markdown table over ``columns`` (default: first row keys)."""
    if not rows:
        return "_(no rows)_\n"
    cols = list(columns) if columns is not None else list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(c), digits) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def write_table(ctx: AnalysisContext, section: str, name: str, text: str) -> Path:
    r"""Write markdown ``text`` to ``<tables_dir>/<section>/<name>.md`` and return the path."""
    target_dir = Path(ctx.tables_dir) / section
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{name}.md"
    target.write_text(text, encoding="utf-8")
    return target


def to_jsonable(obj: Any) -> Any:
    r"""Recursively convert numpy/pathlib values to strict-JSON types (non-finite floats -> None)."""
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path: Path, obj: Any) -> Path:
    r"""Atomically write ``obj`` as strict, indented JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=destination.parent,
                                     prefix=f".{destination.stem}.", suffix=".tmp",
                                     delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(to_jsonable(obj), handle, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, destination)
    return destination
