r"""Publication-quality figures for VFE_3.0 diagnostics (matplotlib; UMAP / networkx).

Figure generators over beliefs (means / gauge frames), attention, covariance, and training
trajectories. Each returns a matplotlib Figure and optionally saves it; colorblind-safe
palette and journal-ish defaults via ``set_publication_style``. The heavier dependencies
(UMAP, networkx) are imported lazily inside the function that needs them, so
the module imports even where one is absent (the function raises a clear message instead).
Tensors are accepted as torch or numpy; everything is detached to numpy for plotting.

A registry (``register_figure``) lets a new figure slot in by name.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from threading import Event, RLock
from types import TracebackType
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")                                            # non-interactive (headless / tests)
import matplotlib.figure
import matplotlib.pyplot as plt

from vfe3.families.base import get_functional_display
from vfe3.metric_contracts import scaling_min_testable_sizes
from vfe3.process_utils import ProcessTree, spawn_process_tree
import numpy as np
from matplotlib.collections import PathCollection, QuadMesh
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter

from vfe3.viz.text import installed_multilingual_sans_serif, token_label

# Wong colorblind-safe qualitative palette (used module-wide, incl. the trajectory defaults).
_CB = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]

def _np(x) -> np.ndarray:
    """Detach a torch tensor (or pass an array) to a contiguous numpy array.

    NumPy has no bfloat16 dtype, so a tensor left in bf16 by an ``amp_dtype`` autocast forward
    cannot cross this boundary at all (``.numpy()`` raises "Got unsupported ScalarType BFloat16").
    Widen it to float32 first -- bf16 is a truncated float32, so the values are preserved exactly.
    """
    if hasattr(x, "detach"):
        import torch
        if x.dtype is torch.bfloat16:
            x = x.float()
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def set_publication_style() -> None:
    """Colorblind-safe palette + journal-ish matplotlib defaults (call once)."""
    try:
        import seaborn as sns
        sns.set_palette("colorblind")
    except ImportError:
        pass
    font_families = installed_multilingual_sans_serif()
    plt.rcParams.update({
        "figure.dpi":      200,
        "savefig.dpi":     300,
        "savefig.bbox":    "tight",
        "font.size":       10,
        "font.family":     font_families,
        "font.sans-serif": font_families,
        "axes.titlesize":  11,
        "axes.labelsize":  10,
        "axes.grid":       True,
        "grid.alpha":      0.25,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


def _figure_target_path(path: 'str | Path') -> Path:
    """Return Matplotlib's effective destination, including its default suffix."""
    target = Path(path)
    if not target.suffix:
        default_format = str(plt.rcParams["savefig.format"]).lstrip(".")
        target = target.with_suffix(f".{default_format}")
    return target


def _plain_tick(value: float, _pos: object = None) -> str:
    r"""Plain decimal tick label (0.3, 20, 1500) for a log axis spanning at most three decades."""
    v = float(value)
    if not np.isfinite(v) or v <= 0.0:
        return ""
    if 1e4 <= v < 1e9:
        return _count_label(v)                                   # 20000 -> '20k'
    if 1e-3 <= v < 1e4:
        text = f"{v:.3g}"
        return f"{v:.0f}" if "e" in text else text
    exponent = int(np.floor(np.log10(v)))
    mantissa = v / 10.0 ** exponent
    if abs(mantissa - 1.0) < 1e-6:
        return rf"$10^{{{exponent}}}$"
    return rf"${mantissa:.2g}\times10^{{{exponent}}}$"


def _tidy_log_axis(axis: object) -> None:
    r"""Readable ticks on one log-scaled axis.

    Matplotlib labels a log axis at decades and, over a narrow range, at minor ticks in
    :math:`a\times10^b` form, so a panel spanning 0.3 to 0.5 reads "3x10^-1, 4x10^-1" and one spanning
    130 to 1800 carries a single "10^3" label. Below 0.6 decade the axis gets linear-step ticks, up to
    1.5 decades ticks at 1, 2, 3, 5 per decade, up to three decades at 1, 2, 5, all with plain decimal
    labels; wider axes keep the decade labels. Minor ticks stay unlabeled throughout.
    """
    lo, hi = sorted(float(v) for v in axis.get_view_interval())
    if not (np.isfinite(lo) and np.isfinite(hi) and lo > 0.0 and hi > lo):
        return
    axis.set_minor_formatter(NullFormatter())
    decades = float(np.log10(hi / lo))
    if decades > 3.0:
        return
    if decades < 0.6:
        axis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
    elif decades < 1.5:
        axis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 5.0)))
    else:
        axis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
    axis.set_major_formatter(FuncFormatter(_plain_tick))


# Legend positions tried when a legend overlaps data, in order of preference.
_LEGEND_LOCATIONS = ("upper right", "upper left", "lower right", "lower left",
                     "center right", "center left", "upper center", "lower center")
_LEGEND_MAX_HEADROOM = 0.5               # largest axes fraction a legend may claim above the data


def _box_overlap(
    ax:       object,
    box:      object,                    # display-coordinate Bbox
    renderer: object,

    *,
    exclude:  Sequence[object] = (),     # artists not counted (the one being placed)
) -> float:
    r"""How much of the axes content a display box covers.

    A drawn line, marker set, scatter or hexbin cloud, bar, or text that the box touches costs 1; a
    translucent band (``fill_between``) costs 0.25, so a box over a band is preferred to one over
    a curve. The box is grown by 2 pt first, so a marker just outside it still counts.
    """
    from matplotlib.path import Path as MplPath

    box = box.padded(renderer.points_to_pixels(2.0))
    cost = 0.0
    for line in ax.get_lines():
        if not line.get_visible():
            continue
        xy = line.get_transform().transform(line.get_xydata())
        xy = xy[np.isfinite(xy).all(axis=1)]
        if not xy.size:
            continue
        if line.get_linestyle() in ("None", "none", "", " ") or xy.shape[0] == 1:
            hit = box.count_contains(xy) > 0
        else:
            hit = MplPath(xy).intersects_bbox(box, filled=False)
        cost += 1.0 if hit else 0.0
    for collection in ax.collections:
        if not collection.get_visible():
            continue
        offsets = np.asarray(collection.get_offsets())
        if isinstance(collection, PathCollection) or offsets.shape[0] > 1:
            points = collection.get_offset_transform().transform(offsets)
            cost += 1.0 if box.count_contains(points[np.isfinite(points).all(axis=1)]) else 0.0
        else:
            transform = collection.get_transform()
            if any(transform.transform_path(path).intersects_bbox(box, filled=True)
                   for path in collection.get_paths()):
                cost += 0.25
    for artist in list(ax.patches) + list(ax.texts):
        if (artist.get_visible() and artist not in exclude
                and artist.get_window_extent(renderer).overlaps(box)):
            cost += 1.0
    return cost


def _legend_overlap(ax: object, legend: object, renderer: object) -> float:
    r""":func:`_box_overlap` of the legend's current box."""
    return _box_overlap(ax, legend.get_window_extent(renderer), renderer)


def _best_legend_location(ax: object, legend: object, renderer: object) -> tuple:
    r"""``(cost, location)`` of the least-overlapping :data:`_LEGEND_LOCATIONS` entry (first zero wins);
    the legend is left at that location."""
    best_cost, best_location = np.inf, legend._loc
    for location in _LEGEND_LOCATIONS:
        legend.set_loc(location)
        cost = _legend_overlap(ax, legend, renderer)
        if cost < best_cost:
            best_cost, best_location = cost, location
        if cost == 0.0:
            break
    legend.set_loc(best_location)
    return best_cost, best_location


def _place_legend(ax: object, legend: object, renderer: object) -> None:
    r"""Move an in-axes legend off the data it overlaps.

    The legend keeps its position when that overlaps nothing. Otherwise the first overlap-free
    position of :data:`_LEGEND_LOCATIONS` is taken. A single-column legend of four or more entries
    that still overlaps is set in two columns when those fit the axes width. When every position
    overlaps a curve, the y range is extended upward by the legend height (at most
    :data:`_LEGEND_MAX_HEADROOM` of the axes) so the legend sits above the data. A legend anchored
    outside the axes by its caller is left alone.
    """
    if getattr(legend, "_bbox_to_anchor", None) is not None:
        return
    if _legend_overlap(ax, legend, renderer) == 0.0:
        return
    best_cost, best_location = _best_legend_location(ax, legend, renderer)
    if best_cost == 0.0:
        return
    if len(legend.get_texts()) >= 4 and getattr(legend, "_ncols", 1) == 1:
        legend.set_ncols(2)
        if legend.get_window_extent(renderer).width <= 0.98 * ax.bbox.width:
            best_cost, best_location = _best_legend_location(ax, legend, renderer)
            if best_cost == 0.0:
                return
        else:
            legend.set_ncols(1)
            legend.set_loc(best_location)
    scale = ax.get_yscale()
    if best_cost < 1.0 or scale not in ("linear", "log") or ax.yaxis_inverted():
        return
    height = legend.get_window_extent(renderer).height / max(ax.bbox.height, 1.0)
    headroom = height + 0.04
    if headroom > _LEGEND_MAX_HEADROOM:
        return
    y0, y1 = ax.get_ylim()
    if scale == "log":
        if y0 <= 0.0 or y1 <= 0.0:
            return
        top = 10.0 ** (np.log10(y0) + (np.log10(y1) - np.log10(y0)) / (1.0 - headroom))
    else:
        top = y0 + (y1 - y0) / (1.0 - headroom)
    ax.set_ylim(y0, top)
    raised_cost, raised_location = np.inf, best_location
    for location in ("upper right", "upper left", "upper center"):
        legend.set_loc(location)
        cost = _legend_overlap(ax, legend, renderer)
        if cost < raised_cost:
            raised_cost, raised_location = cost, location
    if raised_cost < best_cost:
        legend.set_loc(raised_location)
    else:
        ax.set_ylim(y0, y1)
        legend.set_loc(best_location)


def tidy_figure(fig: object) -> object:
    r"""Final layout pass applied to every saved figure.

    Grid lines are removed from image axes (a heatmap's cells, not its coordinates, carry the
    data), in-axes legends are moved off the data they overlap (:func:`_place_legend`), and log
    axes get readable tick labels (:func:`_tidy_log_axis`). The figure is laid out once so the
    legend and data boxes are in their final display positions.
    """
    for ax in fig.axes:
        if ax.images or any(isinstance(c, QuadMesh) for c in ax.collections):
            ax.grid(False)
    fig.draw_without_rendering()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is not None and legend.get_visible() and not ax.images:
            _place_legend(ax, legend, renderer)
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            if axis.get_scale() == "log":
                _tidy_log_axis(axis)
        _unclip_edge_markers(ax)
    return fig


def _unclip_edge_markers(ax: object) -> None:
    r"""Draw markers on the axes edge whole.

    A step axis ends exactly at the first and last logged step, so their markers are cut in half by
    the spines. A marker line or scatter whose data all lie inside the view limits cannot reach
    beyond them, so it is drawn unclipped.
    """
    (x0, x1), (y0, y1) = sorted(ax.get_xlim()), sorted(ax.get_ylim())

    def inside(xy: np.ndarray) -> bool:
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        xy = xy[np.isfinite(xy).all(axis=1)]
        tol_x, tol_y = 1e-9 * max(abs(x0), abs(x1), 1.0), 1e-9 * max(abs(y0), abs(y1), 1e-300)
        return bool(xy.size) and bool(((xy[:, 0] >= x0 - tol_x) & (xy[:, 0] <= x1 + tol_x)
                                       & (xy[:, 1] >= y0 - tol_y) & (xy[:, 1] <= y1 + tol_y)).all())

    for line in ax.get_lines():
        if (line.get_marker() not in (None, "None", "none", "", " ")
                and line.get_transform() == ax.transData and inside(line.get_xydata())):
            line.set_clip_on(False)
    for collection in ax.collections:
        if (isinstance(collection, PathCollection)
                and collection.get_offset_transform() == ax.transData
                and inside(collection.get_offsets())):
            collection.set_clip_on(False)


def _save(fig, path: Optional[str]):
    if path is not None:
        if isinstance(fig, matplotlib.figure.Figure):
            try:
                tidy_figure(fig)
            except Exception as exc:                             # layout polish never costs the figure
                import logging
                logging.getLogger(__name__).warning("tidy_figure skipped for %s (%s)", path, exc)
        target = _figure_target_path(path)
        temporary = target.with_name(
            f".{target.stem}.{uuid4().hex}.tmp{target.suffix}"
        )
        try:
            fig.savefig(str(temporary))
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise RuntimeError("figure renderer did not write a nonempty temporary output")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    return fig


# Page geometry of the figures rebuilt on 2026-09-28 (audit k160 Section 4): at most one two-column
# text width, and no text below 6 pt at that width. The compact rc below is applied while those
# figures are built AND saved, so draw-time tick labels inherit it.
MAX_FIGURE_WIDTH_IN = 7.2
MIN_TEXT_PT = 6.0
_COMPACT_RC: Dict[str, object] = {
    "font.size":        7.0,
    "axes.titlesize":   7.5,
    "axes.labelsize":   7.0,
    "xtick.labelsize":  6.5,
    "ytick.labelsize":  6.5,
    "legend.fontsize":  6.0,
    "figure.titlesize": 8.5,
}


class FigureSkipped(RuntimeError):
    r"""Raised by a figure builder whose figure is route-trivial or has no data to show.

    The message is the logged skip reason. Nothing is written: a caller that catches this logs the
    reason and moves on, and a caller that treats every exception as a failed figure still logs the
    reason text. Retired placeholders (a text box saying "unavailable") are no longer drawn.
    """


def has_estep_iterates(cfg: object) -> bool:
    r"""Whether the legacy shared-objective E-step figure family applies.

    These figures trace the legacy shared-objective E-step. Predictive attention is a readout;
    projected-gradient receiver iterations have a frozen incoming router and a separate receiver
    objective, so the legacy trajectory and objective probes do not apply to either retained mode.
    """
    return False


def route_figure_skip_reasons(cfg: object) -> Dict[str, str]:
    r"""Skip reasons for figures that are identically trivial on the configured route.

    Returns ``{figure_name: reason}``; a name absent from the result is emitted normally. The single
    source of the route gates of audit k160 Section 4, used by :mod:`vfe3.viz.report` and available to
    the training-time emitters (``attention_estep``, ``estep_*``, ``belief_condition``).

    * ``holonomy_curvature`` always: the projected routes run flat transport, whose holonomy is the
      identity by construction, so the figure plots float32 roundoff.
    * The legacy E-step family when :func:`has_estep_iterates` is false: the retained routes use
      predictive readout or a frozen-router receiver objective rather than that shared objective.
    * ``belief_spectrum``, ``spd_ellipses`` and ``belief_condition`` when the covariance is frozen at
      encode (:func:`vfe3.metrics.covariance_frozen_at_encode`): they would describe the decayed
      token table, not beliefs.
    """
    from vfe3 import metrics
    reasons: Dict[str, str] = {}
    reasons["holonomy_curvature"] = (
        "flat transport: holonomy is the identity by construction, so the figure would "
        "plot float32 roundoff")
    if not has_estep_iterates(cfg):
        reason = (f"attention_mode={getattr(cfg, 'attention_mode', 'projected_predictive')!r}: "
                  "legacy shared-objective E-step probes do not apply")
        for name in ("iterate_trajectory", "belief_trajectories",
                     "estep_character", "estep_quality", "estep_grad_norm_decomposition",
                     "attention_estep"):
            reasons[name] = reason
    if metrics.covariance_frozen_at_encode(cfg):
        reason = ("covariance frozen at encode: Sigma is the input-token table row, not a belief "
                  "covariance")
        for name in ("belief_spectrum", "spd_ellipses", "belief_condition"):
            reasons[name] = reason
    return reasons


# ---------------------------------------------------------------------------
# Run-directory table readers for the history figures (metrics.csv and the
# long-format diagnostics/*.csv files written at eval cadence).
# ---------------------------------------------------------------------------

def read_metrics_columns(path: 'str | Path') -> Optional[Dict[str, np.ndarray]]:
    r"""Read ``metrics.csv`` as ``{column: float64 array}``; blank or non-numeric cells are NaN.

    Returns None when the file is absent or has no data rows. Rows keep their file order, so an
    eval-cadence column is NaN on the log-cadence rows between evaluations.
    """
    import csv
    path = Path(path)
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    columns: Dict[str, np.ndarray] = {}
    for key in rows[0]:
        if key is None:
            continue
        values = np.full(len(rows), np.nan)
        for index, row in enumerate(rows):
            raw = row.get(key)
            if raw is None or raw == "":
                continue
            try:
                values[index] = float(raw)
            except ValueError:
                pass
        columns[key] = values
    return columns


def read_long_table(path: 'str | Path') -> Optional[Dict[str, np.ndarray]]:
    r"""Read a long-format diagnostics CSV (``diagnostics/residual_ratio.csv``,
    ``diagnostics/attention_stats.csv``) as ``{column: float64 array}``; None when absent or empty."""
    return read_metrics_columns(path)


def _compact_style():
    r"""Context manager applying :data:`_COMPACT_RC` (7.2 in figures, text >= 6 pt)."""
    return plt.rc_context(_COMPACT_RC)


def _zoom_bar_value_axis(ax, values, *, vertical: bool = True,
                         lo_frac: float = 0.25, hi_frac: float = 0.12) -> None:
    r"""Zoom a bar chart's VALUE axis to a padded ``[min, max]`` window (not ``[0, max]``) so
    near-equal bars are distinguishable instead of all reading as full bars from zero. No-op unless
    there are at least two finite values that are not all equal (so a single- or constant-valued
    chart keeps its default autoscale)."""
    v = np.asarray([x for x in values if x is not None], dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return
    vmin, vmax = float(v.min()), float(v.max())
    if vmax <= vmin:
        return
    span = vmax - vmin
    lo, hi = vmin - lo_frac * span, vmax + hi_frac * span
    (ax.set_ylim if vertical else ax.set_xlim)(lo, hi)


# Default UMAP hyper-parameters of umap_embed (vfe3.semantics passes its own n_neighbors).
_UMAP_N_NEIGHBORS = 15                                           # local-neighborhood size (callers scale with N)
_UMAP_MIN_DIST    = 0.1                                          # display-tuned spacing (0.0 for clustering runs)

# Worker source for the ISOLATED UMAP embedding subprocess (see UMAPWorker).
# init="pca" skips UMAP's spectral eigensolver, which on disconnected / near-degenerate belief
# clouds fails to converge (tiny eigengap) and silently falls back to uniform-random init; PCA is
# deterministic and data-aware. n_jobs=1 matches what random_state already forces, so it also
# silences UMAP's "n_jobs overridden" warning. Same compute, no warning cascade.
_UMAP_WORKER_SRC = (
    "import json, os, sys, traceback\n"
    "import numpy as np\n"
    "import umap\n"
    "for line in sys.stdin:\n"
    "    request = json.loads(line)\n"
    "    try:\n"
    "        X = np.load(request['input'])\n"
    "        reducer = umap.UMAP(n_neighbors=int(request['n_neighbors']),\n"
    "                            min_dist=float(request['min_dist']),\n"
    "                            n_components=int(request['n_components']), init='pca',\n"
    "                            random_state=int(request['seed']), n_jobs=1)\n"
    "        np.save(request['output'], reducer.fit_transform(X))\n"
    "        response = {'ok': True}\n"
    "    except BaseException:\n"
    "        response = {'ok': False, 'error': traceback.format_exc()[-4000:]}\n"
    "    status_tmp = request['status'] + '.tmp'\n"
    "    with open(status_tmp, 'w', encoding='utf-8') as handle:\n"
    "        json.dump(response, handle)\n"
    "    os.replace(status_tmp, request['status'])\n"
)


# Cap on concurrent UMAP interpreters. Each fit is single-threaded by construction (random_state
# forces it), so the pool is the ONLY parallelism available and one core per fit is the right unit.
_UMAP_MAX_WORKERS = 8
_STDERR_UNLINK_POLL_SECONDS = 0.005      # Windows handle-release lag after a drained process tree


def _numba_cache_dir() -> Optional[str]:
    r"""A PERSISTENT numba cache for the UMAP workers; None means "inherit numba's own default".

    The worker used to point ``NUMBA_CACHE_DIR`` at its request tempdir, which ``close`` then
    removed, so umap-learn's numba kernels were JIT-compiled from scratch on every report and the
    cache never survived a single run. An explicitly set ``NUMBA_CACHE_DIR`` in the parent
    environment wins; otherwise the first writable candidate is used.
    """
    import os
    import tempfile
    inherited = os.environ.get("NUMBA_CACHE_DIR")
    if inherited:
        return inherited
    candidates = (Path.home() / ".cache" / "vfe3" / "numba",
                  Path(tempfile.gettempdir()) / "vfe3-numba-cache")
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return str(candidate)
        except OSError:
            continue
    return None


@dataclass(eq=False)
class _UMAPWorkerRecord:
    """Identity-bearing lifecycle state for one owned worker process."""

    tree: ProcessTree
    stderr_handle: object
    identity: str = field(default_factory=lambda: uuid4().hex)
    retiring: bool = False
    retired: bool = False
    requests_submitted: int = 0
    retired_event: Event = field(default_factory=Event)
    cleanup_status: Optional[Dict[str, object]] = None

    @property
    def process(self):
        return self.tree.process


class UMAPWorker:
    r"""A lazily started, crash-isolated POOL of UMAP interpreters reusable within a report.

    One process was enough while embeddings were requested one at a time. The controlled multi-seed
    protocol requests five fits of the SAME features per channel and fifteen per report, and each is
    single-threaded, so a single worker left 23 of 24 cores idle while UMAP accounted for nearly all
    of the figure wall clock. Processes are started on demand up to ``max_workers`` and reused, so a
    caller that only ever asks for one embedding still pays for exactly one interpreter.
    """

    def __init__(
        self,

        *,
        timeout:         float         = 1200.0,
        cleanup_timeout: float         = 5.0,
        max_workers:     Optional[int] = None,
    ) -> None:
        if timeout <= 0.0:
            raise ValueError("UMAP worker timeout must be positive")
        if cleanup_timeout <= 0.0:
            raise ValueError("UMAP cleanup timeout must be positive")
        cores = os.cpu_count() or 1
        self.timeout = timeout
        self.cleanup_timeout = cleanup_timeout
        self.max_workers = (max(1, min(_UMAP_MAX_WORKERS, cores - 1)) if max_workers is None
                            else max(1, int(max_workers)))
        self._counter = 0
        self._workers: List[_UMAPWorkerRecord] = []
        self._lifecycle_lock = RLock()
        self._closing = False
        self._request_paths: set[str] = set()
        self.cleanup_statuses: List[Dict[str, object]] = []
        self._workdir = None

    def _record_cleanup_failure(self, *, reason: str, path: str, exc: BaseException) -> dict:
        status: Dict[str, object] = {
            "pid": None,
            "reason": reason,
            "terminated": False,
            "reaped": False,
            "root_reaped": False,
            "tree_termination_confirmed": False,
            "returncode": None,
            "cleanup_error": f"{type(exc).__name__}: {exc}",
            "elapsed_s": 0.0,
            "path": path,
        }
        with self._lifecycle_lock:
            self.cleanup_statuses.append(status)
        return status

    def _cleanup_request_path(self, path: str) -> None:
        try:
            os.remove(path)
        except FileNotFoundError:
            with self._lifecycle_lock:
                self._request_paths.discard(path)
        except OSError as exc:
            self._record_cleanup_failure(reason="request_cleanup", path=path, exc=exc)
        else:
            with self._lifecycle_lock:
                self._request_paths.discard(path)

    def _ensure(self, count: int) -> None:
        r"""Guarantee at least ``count`` live interpreters (never more than ``max_workers``)."""
        import subprocess
        import sys
        import tempfile
        with self._lifecycle_lock:
            if self._closing:
                raise RuntimeError("UMAP worker pool is closing")
            if self._workdir is None:
                self._workdir = tempfile.mkdtemp(prefix="vfe3_umap_")
            cache_dir = _numba_cache_dir()
            while len(self._workers) < min(count, self.max_workers):
                worker_env = os.environ.copy()
                if cache_dir is None:
                    worker_env.pop("NUMBA_CACHE_DIR", None)
                else:
                    worker_env["NUMBA_CACHE_DIR"] = cache_dir
                handle = open(os.path.join(self._workdir, f"stderr_{uuid4().hex}.log"), "w+b")
                try:
                    tree = spawn_process_tree(
                        [sys.executable, "-c", _UMAP_WORKER_SRC],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=handle,
                        text=True,
                        env=worker_env,
                    )
                except BaseException:
                    handle.close()
                    raise
                self._workers.append(_UMAPWorkerRecord(tree=tree, stderr_handle=handle))

    def _error_tail(self, record: _UMAPWorkerRecord) -> str:
        stderr_handle = record.stderr_handle
        try:
            stderr_handle.flush()
        except (OSError, ValueError):
            return ""
        try:
            with open(stderr_handle.name, "rb") as handle:
                handle.seek(0, os.SEEK_END)
                handle.seek(max(0, handle.tell() - 1000), os.SEEK_SET)
                return handle.read().decode(errors="replace")
        except (OSError, ValueError):
            return ""

    def _retire(
        self,
        record: _UMAPWorkerRecord,
        *,
        reason: str,
    ) -> Optional[Dict[str, object]]:
        """Retire one identity exactly once, without holding the lock while waiting."""
        with self._lifecycle_lock:
            if record.retired:
                return record.cleanup_status
            if record.retiring:
                owner = False
            else:
                record.retiring = True
                owner = True
        if not owner:
            record.retired_event.wait(self.cleanup_timeout * 2.0 + 0.1)
            return record.cleanup_status

        from time import monotonic
        retirement_started = monotonic()

        proc = record.process
        handle = record.stderr_handle
        cleanup_errors: List[str] = []
        returncode = proc.poll()
        status: Dict[str, object] = {
            "pid": int(proc.pid),
            "reason": reason,
            "terminated": True,
            "reaped": returncode is not None,
            "root_reaped": returncode is not None,
            "tree_termination_confirmed": False,
            "returncode": returncode,
            "cleanup_error": None,
            "elapsed_s": 0.0,
        }
        try:
            try:
                status = record.tree.terminate(reason=reason, timeout=self.cleanup_timeout)
            except BaseException as exc:
                cleanup_errors.append(f"{type(exc).__name__}: {exc}")
                returncode = proc.poll()
                status.update(
                    reaped=returncode is not None,
                    root_reaped=returncode is not None,
                    returncode=returncode,
                )

            # Only after bounded tree termination: close streams on daemon helpers sharing the
            # remaining cleanup budget. TextIOWrapper.close() may flush behind a concurrent submit.
            stdin = proc.stdin
            proc.stdin = None
            try:
                stderr_path = handle.name
            except BaseException as exc:
                cleanup_errors.append(f"stderr path: {type(exc).__name__}: {exc}")
                stderr_path = None

            from threading import Thread
            close_deadline = monotonic() + max(
                0.0,
                self.cleanup_timeout - float(status.get("elapsed_s", 0.0)),
            )
            pending_closes = []
            for label, stream in (("stdin", stdin), ("stderr", handle)):
                if stream is None:
                    continue
                completed = Event()
                close_errors: List[BaseException] = []

                def _close_stream(
                    owned_stream=stream,
                    errors=close_errors,
                    done=completed,
                ) -> None:
                    try:
                        owned_stream.close()
                    except BaseException as exc:
                        errors.append(exc)
                    finally:
                        done.set()

                closer = Thread(
                    target=_close_stream,
                    name=f"umap-{label}-close-{proc.pid}",
                    daemon=True,
                )
                try:
                    closer.start()
                except BaseException as exc:
                    cleanup_errors.append(f"{label} close start: {type(exc).__name__}: {exc}")
                    continue
                pending_closes.append((label, completed, close_errors))

            for label, completed, close_errors in pending_closes:
                remaining = max(0.0, close_deadline - monotonic())
                if not completed.wait(remaining):
                    cleanup_errors.append(f"{label} close exceeded cleanup budget")
                elif close_errors:
                    exc = close_errors[0]
                    cleanup_errors.append(f"{label} close: {type(exc).__name__}: {exc}")

            if stderr_path is not None:
                # Windows releases a terminated tree's INHERITED handles a few milliseconds after the
                # job reports itself drained (the gated launcher hands its stderr to the real worker,
                # so the log outlives both). A single unlinked attempt therefore lost the race and
                # published WinError 32 on every worker. Retry inside the caller's cleanup budget --
                # the same policy close() already applies to the scratch directory itself.
                remove_deadline = retirement_started + self.cleanup_timeout
                while True:
                    try:
                        os.remove(stderr_path)
                    except FileNotFoundError:
                        break
                    except OSError as exc:
                        remaining = remove_deadline - monotonic()
                        if remaining <= 0.0:
                            cleanup_errors.append(f"{type(exc).__name__}: {exc}")
                            break
                        Event().wait(min(_STDERR_UNLINK_POLL_SECONDS, remaining))
                    else:
                        break
        finally:
            status["elapsed_s"] = monotonic() - retirement_started
            existing_error = status.get("cleanup_error")
            status["cleanup_error"] = " | ".join(
                part for part in (existing_error, *cleanup_errors) if part
            ) or None
            with self._lifecycle_lock:
                self._workers[:] = [
                    active for active in self._workers if active.identity != record.identity
                ]
                record.cleanup_status = status
                record.retired = True
                record.retiring = False
                self.cleanup_statuses.append(status)
                record.retired_event.set()
        return status

    def _submit(self, record: _UMAPWorkerRecord, request: Dict[str, object]) -> None:
        import json
        from threading import Thread
        proc = record.process
        stream = proc.stdin
        if stream is None:
            raise OSError("UMAP worker input is unavailable")
        payload = json.dumps(request) + "\n"
        completed = Event()
        outcome: List[BaseException] = []

        def _write_request() -> None:
            try:
                stream.write(payload)
                stream.flush()
            except BaseException as exc:
                outcome.append(exc)
            finally:
                completed.set()

        writer = Thread(target=_write_request, name=f"umap-submit-{proc.pid}", daemon=True)
        writer.start()
        if not completed.wait(self.timeout):
            self._retire(record, reason="submit_timeout")
            raise TimeoutError(f"UMAP worker request submission exceeded {self.timeout:g} seconds")
        if outcome:
            exc = outcome[0]
            tail = self._error_tail(record)
            self._retire(record, reason="pipe_closed")
            if "ModuleNotFoundError" in tail and "umap" in tail:
                raise ImportError("umap_embed needs umap-learn (pip install umap-learn)") from exc
            if isinstance(exc, (BrokenPipeError, OSError, ValueError)):
                raise OSError(f"UMAP worker pipe closed: {tail[-500:]}") from exc
            raise exc
        with self._lifecycle_lock:
            if record.retired or record.retiring:
                raise OSError("UMAP worker retired during request submission")
            record.requests_submitted += 1

    def _collect(self, record: _UMAPWorkerRecord, status: str, fout: str) -> np.ndarray:
        r"""Block on one submitted request. The timeout starts HERE, matching the serial contract.

        A queued request (more seeds than workers) therefore gets its full timeout from the moment
        it is awaited rather than sharing one budget with the whole batch.
        """
        import json
        from threading import Thread
        proc = record.process
        ready = Event()
        canceled = Event()

        def _watch() -> None:
            while not canceled.is_set():
                if os.path.exists(status) or proc.poll() is not None:
                    ready.set()
                    return
                canceled.wait(0.01)

        watcher = Thread(target=_watch, name=f"umap-status-{proc.pid}", daemon=True)
        watcher.start()
        with self._lifecycle_lock:
            first_request = record.requests_submitted == 1
        startup_grace = min(0.25, self.cleanup_timeout) if first_request else 0.0
        finished = ready.wait(self.timeout + startup_grace)
        canceled.set()
        watcher.join(timeout=self.cleanup_timeout)
        if not finished and not os.path.exists(status):
            self._retire(record, reason="timeout")
            raise TimeoutError(f"UMAP worker exceeded {self.timeout:g} seconds")
        if not os.path.exists(status):
            returncode = proc.poll()
            tail = self._error_tail(record)
            self._retire(record, reason="worker_exit")
            if "ModuleNotFoundError" in tail and "umap" in tail:
                raise ImportError("umap_embed needs umap-learn (pip install umap-learn)")
            raise OSError(f"UMAP worker exited (rc={returncode}): {tail[-500:]}")
        with open(status, encoding="utf-8") as handle:
            response = json.load(handle)
        if not response.get("ok"):
            error = str(response.get("error", "unknown worker failure"))
            if "ModuleNotFoundError" in error and "umap" in error:
                raise ImportError("umap_embed needs umap-learn (pip install umap-learn)")
            raise OSError(f"UMAP embedding worker failed: {error[-500:]}")
        return np.load(fout)

    def embed(
        self,
        features: np.ndarray,

        *,
        n_neighbors:  int,
        min_dist:     float,
        n_components: int,
        seed:         int,
    ) -> np.ndarray:
        return self.embed_many(features, seeds=(seed,), n_neighbors=n_neighbors,
                               min_dist=min_dist, n_components=n_components)[int(seed)]

    def embed_many(
        self,
        features: np.ndarray,

        *,
        seeds:        Sequence[int],
        n_neighbors:  int,
        min_dist:     float,
        n_components: int,
    ) -> Dict[int, np.ndarray]:
        r"""Embed ONE feature bank under several seeds, one concurrent interpreter per seed.

        Every fit is seeded and single-threaded exactly as the serial path ran it, and the features
        are written once and read by each worker, so the returned coordinates are bit-identical to
        embedding the seeds one at a time. Only the wall clock changes.
        """
        ordered = [int(seed) for seed in seeds]
        if not ordered:
            raise ValueError("embed_many needs at least one seed")
        if len(set(ordered)) != len(ordered):
            raise ValueError(f"embed_many needs distinct seeds, got {ordered!r}")
        self._ensure(len(ordered))
        with self._lifecycle_lock:
            assert self._workdir is not None
            self._counter += 1
            stem = os.path.join(self._workdir, f"request_{self._counter}")
            workers = tuple(self._workers)
        fin = f"{stem}_in.npy"
        np.save(fin, features)
        paths = [fin]
        jobs = []
        try:
            for position, seed in enumerate(ordered):
                record = workers[position % len(workers)]
                fout, status = f"{stem}_{seed}_out.npy", f"{stem}_{seed}_status.json"
                paths += [fout, status, f"{status}.tmp"]
                with self._lifecycle_lock:
                    self._request_paths.update((fout, status, f"{status}.tmp"))
                self._submit(record, dict(input=fin, output=fout, status=status,
                                          n_neighbors=n_neighbors, min_dist=min_dist,
                                          n_components=n_components, seed=seed))
                jobs.append((record, seed, fout, status))
            with self._lifecycle_lock:
                self._request_paths.add(fin)
            return {seed: self._collect(record, status, fout)
                    for record, seed, fout, status in jobs}
        finally:
            for path in paths:
                self._cleanup_request_path(path)

    def close(self) -> List[Dict[str, object]]:
        from time import monotonic
        import shutil
        statuses: List[Dict[str, object]] = []
        with self._lifecycle_lock:
            self._closing = True
            workers = tuple(reversed(self._workers))
            workdir = self._workdir
            self._workdir = None
        for record in workers:
            status = self._retire(record, reason="close")
            if status is not None:
                statuses.append(status)
        with self._lifecycle_lock:
            request_paths = tuple(self._request_paths)
        for path in request_paths:
            self._cleanup_request_path(path)
        if workdir is not None:
            # The numba cache lives OUTSIDE this tree on purpose; only the request scratch goes.
            # Windows can retain a terminated worker's file handles briefly. Retry only inside
            # the caller's explicit cleanup budget and publish the terminal failure, if any.
            cleanup_deadline = monotonic() + self.cleanup_timeout
            while True:
                try:
                    shutil.rmtree(workdir)
                    break
                except FileNotFoundError:
                    break
                except OSError as exc:
                    remaining = cleanup_deadline - monotonic()
                    if remaining <= 0.0:
                        statuses.append(self._record_cleanup_failure(
                            reason="workdir_cleanup", path=workdir, exc=exc,
                        ))
                        break
                    Event().wait(min(0.01, remaining))
        with self._lifecycle_lock:
            self._closing = False
        return statuses

    def __enter__(self) -> "UMAPWorker":
        return self

    def __exit__(
        self,
        _exc_type: Optional[type[BaseException]],
        _exc:      Optional[BaseException],
        _tb:       Optional[TracebackType],
    ) -> None:
        self.close()


def _umap_prepare(
    features,                            # (N, D) tensor/array
    n_neighbors:  int,
    n_components: int,
) -> 'tuple[np.ndarray, int, int, Optional[np.ndarray]]':
    r"""Shared clamping and degenerate-input handling for the two UMAP entry points.

    Returns ``(X, n_neighbors, n_components, degenerate)``; a non-None ``degenerate`` is the layout
    to return instead of fitting.
    """
    X = _np(features)
    # PCA init raises when n_components exceeds the feature dim or N-1 (e.g. the phi channel of a
    # small algebra, tiny CPU test banks), so clamp it the same way n_neighbors is clamped below.
    n_components = max(1, min(n_components, X.shape[1], max(1, X.shape[0] - 1)))
    # A fully collapsed channel (every point identical -> zero variance) has no embedding, and PCA
    # init would divide by total variance 0 and yield NaN. Return a trivial finite layout so the
    # downstream clustering / KDE stay valid (faithful: constant features carry no 2-D structure).
    if X.shape[0] < 3 or float(np.ptp(X, axis=0).max()) <= 0.0:
        return X, n_neighbors, n_components, np.zeros((X.shape[0], n_components), dtype=float)
    return X, min(n_neighbors, max(2, X.shape[0] - 1)), n_components, None


def umap_embed(
    features,                            # (N, D) tensor/array

    *,
    n_neighbors:  int = _UMAP_N_NEIGHBORS,
    min_dist:     float = _UMAP_MIN_DIST,
    n_components: int = 2,
    seed:         int = 0,
    worker:        Optional[UMAPWorker] = None,
) -> np.ndarray:
    r"""UMAP embedding of ``features`` ((N, D) -> (N, n_components)), run in an ISOLATED subprocess.

    umap-learn's numba/llvmlite native layer can die with a Windows ACCESS VIOLATION when it
    initializes inside a long-running, heavily loaded process (observed on Python 3.14 after
    hundreds of tests: llvmlite ``check_jit_execution`` faults; a fresh process imports numba
    fine) -- a process-killing crash no in-process try/except can catch, taking the whole
    training finalize / test session down with it. Running the embedding in a subprocess fully
    isolates the native layer: same computation, same seeded result. A report may reuse one such
    worker for several embeddings without reimporting numba. A failing subprocess (numba genuinely
    unsupported, umap-learn absent) raises the OSError/ImportError the UMAP consumers were already
    written to handle (audit 2026-07-05 verification fix)."""
    X, n_neighbors, n_components, degenerate = _umap_prepare(features, n_neighbors, n_components)
    if degenerate is not None:
        return degenerate
    if worker is not None:
        return worker.embed(X, n_neighbors=n_neighbors, min_dist=min_dist,
                            n_components=n_components, seed=seed)
    with UMAPWorker(max_workers=1) as isolated_worker:
        return isolated_worker.embed(X, n_neighbors=n_neighbors, min_dist=min_dist,
                                     n_components=n_components, seed=seed)


def attention_graph(
    beta,                                # (N, N) attention weights

    *,
    threshold: float = 0.05,
):
    """Build a weighted directed graph from the attention matrix (edges above ``threshold``)."""
    import networkx as nx
    B = _np(beta)
    N = B.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    for i in range(N):
        for j in range(N):
            if i != j and B[i, j] > threshold:
                G.add_edge(i, j, weight=float(B[i, j]))
    return G


def plot_attention_graph(
    beta,                                # (N, N)

    *,
    threshold: float = 0.05,
    path:      Optional[str] = None,
):
    """Draw the attention graph (spring layout, edge width ~ weight)."""
    import networkx as nx
    G = attention_graph(beta, threshold=threshold)
    fig, ax = plt.subplots(figsize=(5, 5))
    pos = nx.spring_layout(G, seed=0)
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=160, node_color="#4C72B0")
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_color="white")
    if weights:
        nx.draw_networkx_edges(G, pos, ax=ax, width=[2.5 * w for w in weights],
                               edge_color="#555555", alpha=0.6, arrowsize=8)
    ax.set_title("Attention graph"); ax.axis("off")
    return _save(fig, path)


def _attn_log_bounds(
    M:     np.ndarray,                   # attention weights (any shape)
    vmin:  Optional[float] = None,
    vmax:  Optional[float] = None,

    *,
    floor: float           = 1e-6,
) -> tuple:                              # (vmin, vmax) valid for a LogNorm scale
    r"""Positive-entry (vmin, vmax) for a log attention scale (scale bottoms out at ``floor`` = 1e-6).

    Causal-masked future positions are exact zeros (softmax over a -inf prior), so only the
    active (positive) entries set the scale. ``vmin`` is floored at ``floor`` (default 1e-6) so the
    log scale resolves attention weights down to 1e-6 while a long causal-softmax tail of near-zero
    weights cannot wash a panel out, and kept strictly below ``vmax`` so a uniform map stays a valid
    LogNorm. Pass both bounds to share one scale across several panels.
    """
    if vmax is None:
        pos = M[M > 0]
        vmax = float(pos.max()) if pos.size else 1.0
    if vmin is None:
        pos = M[M > 0]
        vmin = float(pos.min()) if pos.size else vmax * floor
    vmin = max(vmin, floor)                                       # bottom the log scale at 1e-6
    if vmin >= vmax:                                             # degenerate / uniform map
        vmin = vmax * floor
    return float(vmin), float(vmax)


def _attn_imshow(ax, B: np.ndarray, *, vmin: float, vmax: float, log: bool = True, cmap: str = "magma"):
    r"""imshow one (N, N) attention map. ``log`` (default) uses ``LogNorm`` to resolve the peaky
    off-diagonal structure a linear scale washes to black. On a causal map (every entry above the
    diagonal zero) the future keys are masked and drawn light gray, so they cannot be mistaken for
    scorable keys with negligible weight. On the log scale the colormap's near-black bottom 15% is
    dropped, so the lowest in-range decade stays readable, and every weight below ``vmin`` --
    including an exact zero on or below the diagonal (fp32 underflow) -- is drawn pure black as
    "below the scale floor". ``cmap`` selects the color family (belief beta = 'magma')."""
    base = plt.get_cmap(cmap)
    if log:
        cmap_obj = ListedColormap(base(np.linspace(0.15, 1.0, 256)), name=f"{base.name}_floored")
        cmap_obj.set_under("black")
    else:
        cmap_obj = base.copy()
    cmap_obj.set_bad("#dddddd")
    B = np.asarray(B, dtype=float)
    shown = np.where(B > 0, B, np.finfo(float).tiny if log else 0.0)
    if B.ndim == 2 and B.shape[0] == B.shape[1] and B.shape[0] > 1:
        future = np.triu(np.ones(B.shape, dtype=bool), 1)
        if not np.any(B[future] > 0):
            shown = np.where(future, np.nan, shown)
    if log:
        return ax.imshow(shown, cmap=cmap_obj, aspect="auto", norm=LogNorm(vmin=vmin, vmax=vmax))
    return ax.imshow(shown, cmap=cmap_obj, aspect="auto", vmin=0.0, vmax=vmax)


def plot_attention_heatmap(
    beta,                                # (N, N)

    *,
    log:    bool            = True,
    title:  str             = "Attention",
    cmap:   str             = "magma",   # color family for the heatmap
    symbol: str             = r"\beta",  # colorbar math symbol
    vmin:   Optional[float] = None,
    vmax:   Optional[float] = None,
    path:   Optional[str]   = None,
):
    r"""Log-scaled heatmap of one attention map (rows = queries i, cols = keys j).

    Attention is a peaky causal softmax (most mass on a few keys, exact zeros above the diagonal),
    so the default ``log`` scale (matplotlib ``LogNorm`` on beta) resolves the off-diagonal
    structure a linear scale collapses to black; the causal-masked zeros render as the 'bad'
    color. Pass shared ``vmin``/``vmax`` to make several panels comparable; otherwise the positive
    entries set the scale (log floor at 1e-6; weights below it draw black). ``cmap``/``symbol`` set
    the heatmap color family and the colorbar's math symbol.
    """
    B = _np(beta)
    vlo, vhi = _attn_log_bounds(B, vmin, vmax)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = _attn_imshow(ax, B, vmin=vlo, vmax=vhi, log=log, cmap=cmap)
    label = rf"${symbol}_{{ij}}$ (log scale)" if log else rf"${symbol}_{{ij}}$"
    fig.colorbar(im, ax=ax, shrink=0.8, label=label, extend="min" if log else "neither")
    ax.set(title=title, xlabel="key j", ylabel="query i")
    return _save(fig, path)


def plot_attention_grid(
    maps,                                # (L, H, N, N) per-layer per-head attention (or (H,N,N) / (N,N))

    *,
    log:   bool          = True,
    title: str           = "Attention",
    path:  Optional[str] = None,
):
    """Grid of attention heatmaps: rows = layers, cols = heads (rows query i, cols key j).

    Accepts a per-layer/per-head stack ``(L, H, N, N)`` (as :meth:`VFEModel.attention_maps`
    returns), a single layer ``(H, N, N)``, or a single map ``(N, N)``. A shared LOG color scale
    across all panels (default ``log``) makes heads/layers comparable and resolves the peaky
    off-diagonal structure; ``squeeze=False`` keeps the L==1 / H==1 axes array 2-D so indexing is
    uniform. For one figure per (layer, head) instead of a grid, call :func:`plot_attention_heatmap`
    per panel (as :meth:`RunArtifacts.save_attention_maps` does).
    """
    M = _np(maps)
    if M.ndim == 2:                      # (N, N) -> one layer, one head
        M = M[None, None]
    elif M.ndim == 3:                    # (H, N, N) -> one layer
        M = M[None]
    L, H = M.shape[0], M.shape[1]
    fig, axes = plt.subplots(L, H, figsize=(2.6 * H + 1.0, 2.6 * L + 0.6), squeeze=False)
    vlo, vhi = _attn_log_bounds(M)                                # one scale shared across panels
    im = None
    for li in range(L):
        for hi in range(H):
            ax = axes[li][hi]
            im = _attn_imshow(ax, M[li, hi], vmin=vlo, vmax=vhi, log=log)
            ax.set_xticks([]); ax.set_yticks([])
            if li == 0:
                ax.set_title(f"head {hi}")
            if hi == 0:
                ax.set_ylabel(f"layer {li}\nquery $i$")
            if li == L - 1:
                ax.set_xlabel("key $j$")
    if im is not None:
        label = r"$\beta_{ij}$ (log scale)" if log else r"$\beta_{ij}$"
        fig.colorbar(im, ax=list(axes.ravel()), shrink=0.85, label=label,
                     extend="min" if log else "neither")
    fig.suptitle(title)
    return _save(fig, path)


def plot_covariance_ellipses(
    mu,                                  # (N, K) belief means
    sigma,                               # (N, K) diagonal variances

    *,
    dims:  tuple = (0, 1),
    path:  Optional[str] = None,
):
    """Plot 1-sigma Gaussian ellipses for two belief coordinates (diagonal beliefs)."""
    from matplotlib.patches import Ellipse
    m = _np(mu); s = _np(sigma)
    a, b = dims
    fig, ax = plt.subplots(figsize=(4.5, 4))
    for i in range(m.shape[0]):
        e = Ellipse((m[i, a], m[i, b]),
                    width=2 * np.sqrt(s[i, a]), height=2 * np.sqrt(s[i, b]),
                    alpha=0.3, facecolor="#4C72B0", edgecolor="#26456E")
        ax.add_patch(e)
    ax.scatter(m[:, a], m[:, b], s=10, color="#26456E")
    ax.set(title="Belief covariance ellipses", xlabel=f"dim {a}", ylabel=f"dim {b}")
    ax.autoscale_view()
    return _save(fig, path)


def _rolling_mean(v: np.ndarray, window: int) -> np.ndarray:
    r"""Centered moving average of ``v`` with edge-normalized windows (no zero-padding bias)."""
    if window <= 1 or v.size <= 2:
        return v
    window = int(min(window, v.size))
    kernel = np.ones(window)
    # NaN-aware: average only the FINITE taps in each window, so a single non-positive point (NaN'd
    # for a log-y axis) does not blank a whole window-width segment of the trend (audit r2 id13).
    # All-finite input -> mask all True -> byte-identical to the plain edge-normalized average.
    mask = np.isfinite(v)
    num = np.convolve(np.where(mask, v, 0.0), kernel, mode="same")
    den = np.convolve(mask.astype(v.dtype), kernel, mode="same")  # finite-tap count per position
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def _log_rolling_mean(v: np.ndarray, window: int) -> np.ndarray:
    r"""Centered rolling geometric mean :math:`\exp\bigl(\overline{\log v}\bigr)` for a positive series.

    The trend of a series drawn on a log axis. An arithmetic mean turns one sample ten times its
    neighbors into a window-wide plateau at twice the baseline, a level that was never logged; in log
    space the same sample is a bump of :math:`10^{1/w}`.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.exp(_rolling_mean(np.log(v), window))


def _kstep(v, _pos=None) -> str:
    r"""Compact training-step tick label: 5000 -> '5k', 12500 -> '12.5k' (5-digit step labels
    overrun into an unreadable smear at the trajectory figure width otherwise)."""
    v = float(v)
    if abs(v) >= 1000.0:
        s = v / 1000.0
        return f"{s:.0f}k" if abs(s - round(s)) < 1e-9 else f"{s:.1f}k"
    return f"{v:.0f}"


def _step_xaxis(ax) -> None:
    r"""Uncrowded training-step x-axis: at most ~6 integer ticks rendered via :func:`_kstep`.
    Applied to every step-indexed trajectory so the default ~9 five-digit labels stop colliding."""
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.xaxis.set_major_formatter(FuncFormatter(_kstep))


def _set_finite_xlim(
    ax:     object,
    values: object,
) -> bool:
    """Set finite, nondegenerate x limits; pad a singleton by at least one step."""
    finite = _np(values).reshape(-1).astype(float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return False
    lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        pad = max(1.0, abs(lo) * 0.01)
        lo, hi = lo - pad, hi + pad
    ax.set_xlim(lo, hi)
    return True


def plot_trajectory(
    values,                              # (T,) a scalar series (loss / free energy / diagnostic)
    steps          = None,               # (T,) x-coordinates (real training step); None -> sample index

    *,
    ylabel:         str           = "loss",
    title:          str           = "Training trajectory",
    color:          str           = _CB[0],
    logy:           bool          = False,
    annotate_final: bool          = False,
    median_line:    bool          = False,
    smooth:         int           = 0,
    annotate:       Optional[str] = None,   # 'min' | 'max' -> mark + label the running best point
    epoch_boundaries: Optional[Sequence[float]] = None,
    markers:        Optional[Tuple[Sequence[float], Sequence[float]]] = None,   # (steps, values) of spike rows
    path:           Optional[str] = None,
):
    r"""Line plot of a scalar trajectory over the real training step.

    The x-axis is ``steps`` when given (else the sample index, so the axis is no longer mislabeled
    as a step). A long noisy series (pass ``smooth=w``) draws a faint raw line under a centered
    ``w``-point rolling mean and no per-point markers, which a thousand-point series otherwise
    collapses into a solid band; a short series (<=50 points) keeps markers. ``logy`` is for wide-range
    or heavy-tailed series (perplexity, holonomy). ``annotate='min'|'max'`` marks that extremum with
    its step, ``annotate_final`` tags the last value at the right edge, and ``median_line`` draws the
    series median as a dashed reference (the typical value on a heavy-tailed log axis where the
    spikes sit decades above the floor). ``epoch_boundaries`` adds subdued vertical guides at
    completed corpus passes; it does not alter either the raw series or its smoothing. ``markers``
    ``(steps, values)`` are the divergence-spike rows, drawn as crosses outside the line and its
    smoothing (:func:`_overlay_spike_markers`).
    """
    v = _np(values).reshape(-1).astype(float)
    x = np.arange(v.size, dtype=float) if steps is None else _np(steps).reshape(-1).astype(float)
    vplot = np.where(v > 0, v, np.nan) if logy else v            # log axis drops non-positive entries
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    if smooth and smooth > 1 and v.size > 50:
        ax.plot(x, vplot, lw=0.7, color=color, alpha=0.25)            # faint raw underlay
        ax.plot(x, _rolling_mean(vplot, smooth), lw=1.8, color=color) # rolling-mean trend
    elif v.size <= 50:
        ax.plot(x, vplot, marker="o", ms=3, lw=1.5, color=color)
    else:
        ax.plot(x, vplot, lw=1.2, color=color)
    if epoch_boundaries is not None:
        for boundary in epoch_boundaries:
            if np.isfinite(boundary):
                ax.axvline(float(boundary), color="#777777", ls=":", lw=0.8, alpha=0.45, zorder=0)
    if logy:
        ax.set_yscale("log")
    if markers is not None and _overlay_spike_markers(
            ax, {"step": markers[0], "value": markers[1]}, "value", "#444444", logy=logy):
        title = f"{title} (x: divergence-spike rows)"
    if median_line and v.size:
        med = float(np.median(v))
        ax.axhline(med, color="#444444", ls="--", lw=1)
        ax.annotate(f"median {med:.2g}", xy=(0.985, med), xycoords=("axes fraction", "data"),
                    ha="right", va="bottom", fontsize=7, color="#444444")
    callout = None
    if annotate in ("min", "max") and v.size:
        idx = int(np.argmin(v) if annotate == "min" else np.argmax(v))
        tag = "best" if annotate == "min" else "max"
        ax.scatter([x[idx]], [v[idx]], s=28, color="black", zorder=5)
        # ``:.4g`` (matching annotate_final) so a heavy-tailed log series whose extremum is O(1e-4)
        # prints its real value instead of rounding to "0.0" under ``:.1f``.
        callout = ax.annotate(f"{tag} {v[idx]:.4g}\n@ step {int(x[idx]):,}",
                              xy=(x[idx], v[idx]), xytext=(0, 0), textcoords="offset points",
                              fontsize=7.5, arrowprops=dict(arrowstyle="->", lw=0.8, color="black"))
        finite_x = x[np.isfinite(x)]
        left_half = bool(finite_x.size) and x[idx] < 0.5 * (finite_x.min() + finite_x.max())
    if annotate_final and v.size:
        ax.annotate(f"{v[-1]:.4g}", xy=(x[-1], v[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, fontweight="bold", color=color)
    if steps is not None and v.size and _set_finite_xlim(ax, x):
        _step_xaxis(ax)
    ax.set(title=title, xlabel="training step", ylabel=ylabel)
    fig.tight_layout()
    if callout is not None:
        _place_callout(ax, callout, toward_interior=1 if left_half else -1,
                       upward=annotate == "min")
    return _save(fig, path)


def _place_callout(
    ax:              object,
    callout:         object,             # Annotation with textcoords="offset points"
    *,
    toward_interior: int,                # +1: the point is in the left half, -1: the right half
    upward:          bool,               # True for a minimum (room above it), False for a maximum
) -> None:
    r"""Put an extremum callout where its text stays inside the axes and crosses the fewest artists.

    Candidates, in order: beside the point toward the middle of the x range, then diagonally into
    the plot, then the same on the other side. A box outside the axes is ruled out, since it
    would overrun the tick labels (belief_condition.png, max near the left edge) or the title
    (grad_norm.png, max near the top).
    """
    fig = ax.figure
    fig.draw_without_rendering()
    renderer = fig.canvas.get_renderer()
    vertical = 1 if upward else -1
    near, far = ("bottom", "top") if upward else ("top", "bottom")
    candidates = []
    for side in (toward_interior, -toward_interior):
        ha = "left" if side > 0 else "right"
        candidates += [(14 * side, 4 * vertical, ha, near), (14 * side, -4 * vertical, ha, far),
                       (12 * side, 30 * vertical, ha, near), (40 * side, 30 * vertical, ha, near),
                       (12 * side, 55 * vertical, ha, near), (60 * side, 55 * vertical, ha, near)]
    best = None
    for dx, dy, ha, va in candidates:
        callout.xyann = (dx, dy)
        callout.set_horizontalalignment(ha)
        callout.set_verticalalignment(va)
        box = callout.get_window_extent(renderer)
        inside = (box.x0 >= ax.bbox.x0 and box.x1 <= ax.bbox.x1
                  and box.y0 >= ax.bbox.y0 and box.y1 <= ax.bbox.y1)
        cost = _box_overlap(ax, box, renderer, exclude=(callout,)) + (0.0 if inside else 100.0)
        if best is None or cost < best[0]:
            best = (cost, dx, dy, ha, va)
        if cost == 0.0:
            break
    _, dx, dy, ha, va = best
    callout.xyann = (dx, dy)
    callout.set_horizontalalignment(ha)
    callout.set_verticalalignment(va)


# ---------------------------------------------------------------------------
# Figure registry: name -> generator. New figures slot in by name.
# ---------------------------------------------------------------------------
_FIGURES: Dict[str, Callable] = {
    "attention_graph":     plot_attention_graph,
    "attention_heatmap":   plot_attention_heatmap,
    "attention_grid":      plot_attention_grid,
    "covariance_ellipses": plot_covariance_ellipses,
    "trajectory":          plot_trajectory,
}


def register_figure(name: str, *, override: bool = False) -> Callable:
    """Decorator registering a figure generator under ``name``.

    Duplicate keys fail closed (audit 2026-07-01 round-3): a second registration under an
    existing name silently shadowed the first. Pass ``override=True`` to replace deliberately.
    """
    def _wrap(fn: Callable) -> Callable:
        if name in _FIGURES and not override:
            raise KeyError(f"figure {name!r} already registered; pass override=True to replace")
        _FIGURES[name] = fn
        return fn
    return _wrap


def get_figure(name: str) -> Callable:
    """Return the registered figure generator (KeyError if absent)."""
    if name not in _FIGURES:
        raise KeyError(f"no figure {name!r}; available: {sorted(_FIGURES)}")
    return _FIGURES[name]


# ===========================================================================
# Publication figures (the claim-linked set). Each consumes the vfe3.metrics
# measurements / vfe3.viz.extract runner outputs, builds a multi-panel
# composite, registers by name, and returns the Figure.
# ===========================================================================



def _median_band(
    ax,
    x:     np.ndarray,
    mat:   np.ndarray,                   # (T, M) -> median + 10/90 band over axis 1
    color: str,
    label: str,
    lo:    float = 10.0,
    hi:    float = 90.0,
):
    """Plot the median of ``mat`` over axis 1 with a 10-90 percentile band (single-seed spread)."""
    med = np.median(mat, axis=1)
    ax.plot(x, med, color=color, lw=1.8, label=label)
    ax.fill_between(x, np.percentile(mat, lo, axis=1), np.percentile(mat, hi, axis=1),
                    color=color, alpha=0.2)
    return med


def _ecdf(ax, x, color: str, label: str):
    """Plot the empirical CDF of ``x`` (clamped positive for a log axis)."""
    xs = np.sort(np.clip(_np(x).ravel(), 1e-12, None))
    if xs.size == 0:
        return
    ax.plot(xs, np.arange(1, xs.size + 1) / xs.size, color=color, lw=1.8, label=label)


# Canonical key -> (legend label, multi-line bar name) for the inner alignment-energy components, in the
# order they stack. The data term (CE) is separate and is co-plotted as a held-out reference, never
# summed into this inner diagnostic.
_FE_LABELS = {
    "self":              ("self-coupling",     "self-coupling\n$\\mathrm{KL}(q\\|p)$"),
    "belief":            ("belief coupling",   "belief-coupling\n$\\lambda_\\beta\\sum\\mathrm{KL}(q\\|\\Omega q)$"),
    "attention_entropy": ("attention entropy", "attention-entropy\n$\\lambda_\\beta\\tau\\beta\\log(\\beta/\\pi)$"),
}


def _fe_terms(history: Dict, lambda_beta, *, include_attention_entropy=True) -> tuple:
    r"""Per-eval complexity free-energy components shared by the decomposition/co-descent figures.

    Returns ``(step, components, total, ce)`` where ``components`` is an ordered list of
    ``(canonical_key, array)`` pairs whose sum is ``total`` -- the per-token inner alignment energy
    that the E-step descends. ``total`` does not include a data/likelihood term, so it is not the
    canonical full variational free energy. Cross-entropy is returned separately as ``ce`` for
    co-plotting and is never summed into this diagnostic. The belief blocks carry the (scalar or per-row)
    ``lambda_beta`` weight; the attention-entropy block is included only when
    ``include_attention_entropy`` (matching ``free_energy_terms``' gate on ``total``). The logged
    diagnostics are already per token, so every component is commensurate and ``total`` matches the
    ``inner_alignment_energy_total`` column.
    """
    step = _np(history["step"]).astype(float)
    lb   = _np(lambda_beta).astype(float)
    zeros = np.zeros(step.shape, dtype=float)
    comps = [("self",   _np(history["self_coupling"]).astype(float)),
             ("belief", lb * _np(history["belief_coupling"]).astype(float))]
    if include_attention_entropy:
        comps.append(("attention_entropy", lb * _np(history["attention_entropy"]).astype(float)))
    total = sum((c for _, c in comps), zeros.copy())
    ce = _np(history["val_ce"]).astype(float) if "val_ce" in history else np.full(step.shape, np.nan)
    return step, comps, total, ce


@register_figure("free_energy_codescent")
@register_figure("free_energy_relationship")
def plot_free_energy_relationship(
    history: Dict,                       # step, self_coupling, belief_coupling, attention_entropy, val_ce

    *,
    lambda_beta:               'float | np.ndarray' = 1.0,
    include_attention_entropy: bool  = True,
    divergence_family:         str = "renyi",
    path:                      Optional[str] = None,
):
    r"""Descriptive relationship between the target-blind objective and held-out loss.

    Twin y-axes over the real training step -- the inner alignment energy (self-coupling +
    the lambda_beta-scaled belief-coupling and attention-entropy, left, solid) and the held-out
    validation CE (right, dashed) -- each a faint raw line under a rolling-mean trend. This energy
    contains no observation likelihood and is therefore not the canonical full variational free
    energy; the CE is a separate held-out readout. The final values are tagged and the Pearson
    association of the two curves is in the title; it summarizes co-movement and does not establish
    that changing either quantity causes the other to change.
    """
    step, _comps, total, ce = _fe_terms(history, lambda_beta,
                                        include_attention_entropy=include_attention_entropy)
    objective_label, objective_units = _registered_divergence_axis(divergence_family)
    keep = np.isfinite(total) & np.isfinite(ce)
    step, total, ce = step[keep], total[keep], ce[keep]
    w = max(5, total.size // 80)
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    ax.plot(step, total, lw=0.7, color=_CB[0], alpha=0.25)
    # Final values fold into the legend labels; standalone right-edge annotations otherwise land on top
    # of the opposite (twin) axis tick labels -- the left energy tag overlaps the right CE ticks.
    flab = f"inner energy (left) = {total[-1]:.1f}" if step.size else "inner energy (left)"
    ln1, = ax.plot(step, _rolling_mean(total, w), lw=2.0, color=_CB[0], label=flab)
    ax.set(xlabel="training step", ylabel=_objective_axis_text(
        f"inner {objective_label}", objective_units, per_token=True))
    ax.yaxis.label.set_color(_CB[0]); ax.tick_params(axis="y", colors=_CB[0])
    if step.size and _set_finite_xlim(ax, step):
        _step_xaxis(ax)
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)                         # set_publication_style hides it by default
    ax2.plot(step, ce, lw=0.7, color=_CB[1], alpha=0.25)
    clab = f"val CE (right) = {ce[-1]:.2f}" if step.size else "val CE (right)"
    ln2, = ax2.plot(step, _rolling_mean(ce, w), lw=2.0, color=_CB[1], ls="--", label=clab)
    ax2.set_ylabel("validation CE (nats/token)", color=_CB[1]); ax2.tick_params(axis="y", colors=_CB[1])
    if total.size > 2:
        r = float(np.corrcoef(total, ce)[0, 1])
        ax.set_title(rf"Inner-objective / CE relationship (Pearson $r={r:+.2f}$)")
    else:
        ax.set_title("Inner-objective / CE relationship")
    ax.legend([ln1, ln2], [ln1.get_label(), ln2.get_label()], loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, path)


plot_free_energy_codescent = plot_free_energy_relationship


def _registered_divergence_axis(name: str) -> tuple[str, Optional[str]]:
    """Return typed display metadata, or a unit-free generic divergence fallback."""
    try:
        display = get_functional_display(name)
    except KeyError:
        return "divergence", None
    return display.label, display.units


def _objective_axis_text(label: str, units: Optional[str], *, per_token: bool = False) -> str:
    suffix = "/token" if per_token else ""
    return f"{label} ({units}{suffix})" if units else f"{label}{suffix}"


@register_figure("free_energy_decomposition")
def plot_free_energy_decomposition(
    history: Dict,                       # step, self_coupling, belief_coupling, attention_entropy, val_ce

    *,
    lambda_beta:               'float | np.ndarray' = 1.0,
    include_attention_entropy: bool  = True,
    divergence_family:         str = "renyi",
    path:                      Optional[str] = None,
):
    r"""The per-token inner alignment-energy budget and how its terms move over training.

    Panel A: the F-contributions at the LAST eval on a LOG x-axis with the value past each bar, so the
    dominant self-coupling and the order-of-magnitude-smaller terms are all legible (a linear bar
    collapses the small ones to slivers, the failure mode of the old single bar). Panel B: the same terms
    at the early/mid/late thirds of training on a log y-axis -- self-coupling sits flat near its value
    while the belief-coupling and attention-entropy terms carry the descent. The bars are the inner
    alignment energy (the quantity the E-step minimizes): self-coupling and the lambda_beta-scaled
    belief-coupling and attention-entropy. The data/likelihood term (CE) is not shown, so this
    diagnostic must not be read as the canonical full variational free energy; see the co-descent
    figure for the separate CE.
    """
    step, comps, _total, _ce = _fe_terms(history, lambda_beta,
                                         include_attention_entropy=include_attention_entropy)
    objective_label, objective_units = _registered_divergence_axis(divergence_family)
    dynamic_labels = dict(_FE_LABELS)
    dynamic_labels["self"] = ("self-coupling", f"self-coupling\n{objective_label}(q,p)")
    dynamic_labels["belief"] = (
        "belief coupling", f"belief-coupling\n{objective_label}(q,transported q)")
    names  = [dynamic_labels[k][1] for k, _ in comps]
    labels = [dynamic_labels[k][0] for k, _ in comps]
    series = [c for _, c in comps]
    colors = _CB[:len(series)]
    last = np.array([s[-1] for s in series])
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2))
    # Panel A -- convergence snapshot (largest at top), log x so every term reads despite the ~30x gap.
    y = np.arange(len(series))[::-1]
    axes[0].barh(y, last, color=colors, height=0.62)
    axes[0].set_xscale("log")
    axes[0].set_yticks(y); axes[0].set_yticklabels(names, fontsize=8)
    for yi, val in zip(y, last):
        axes[0].annotate(f"{val:.1f}", xy=(val, yi), xytext=(4, 0), textcoords="offset points",
                         va="center", ha="left", fontsize=9)
    axes[0].set_xlim(float(last.min()) * 0.5, float(last.max()) * 3.0)
    axes[0].set(xlabel=_objective_axis_text(
                    "inner-objective contribution", objective_units, per_token=True) + " (log scale)",
                title=f"Budget at step {int(step[-1]):,}")
    # Panel B -- early/mid/late medians, log y so the flat dominant term and the moving terms coexist.
    thirds  = np.array_split(np.arange(step.size), 3)
    centers = np.arange(3)
    n_bars  = len(series)
    width   = min(0.2, 0.8 / max(n_bars, 1))                         # keep the grouped bars inside one tick
    off0    = (n_bars - 1) / 2.0                                     # center the group regardless of bar count
    for j, arr in enumerate(series):
        meds = [float(np.median(arr[idx])) if idx.size else np.nan for idx in thirds]   # skip empty thirds
        axes[1].bar(centers + (j - off0) * width, meds, width=width, color=colors[j], label=labels[j])
    axes[1].set_yscale("log")
    axes[1].set_ylim(top=axes[1].get_ylim()[1] * 2.2)            # headroom for the legend above the bars
    axes[1].set_xticks(centers); axes[1].set_xticklabels(["early", "mid", "late"])
    axes[1].set(xlabel="training third",
                ylabel=_objective_axis_text(
                    "inner term (median)", objective_units, per_token=True) + " (log scale)",
                title="Inner-objective terms across training thirds")
    axes[1].legend(fontsize=7.5, frameon=False, ncol=2, loc="upper right")
    fig.tight_layout()
    return _save(fig, path)


# The canonical diagnostic (diagnose_canonical_free_energy) on the projected routes evaluates, per token and at
# the beliefs each router read, F_i = alpha_i D(q_i||p_i) + R(alpha_i)
# + lambda_beta sum_h [sum_j beta*_ij D^h_ij + tau_h sum_j beta*_ij log(beta*_ij / pi_ij)] with the canonical
# router beta* (vfe3.free_energy.canonical_routing_rows). No step on these routes optimizes it, so the figures
# below title it as a diagnostic and carry no E-step or co-descent reading (audit 2026-10-10 round 2, #17).
_CANONICAL_FE_LABELS = {
    "self":              ("self-coupling", "self-coupling\n" r"$\alpha_iD(q_i\|p_i)+R(\alpha_i)$"),
    "belief":            ("belief coupling", "belief coupling\n" r"$\lambda_\beta\sum_j\beta^*_{ij}D_{ij}$"),
    "attention_entropy": ("router entropy", "router entropy\n"
                          r"$\lambda_\beta\tau\sum_j\beta^*_{ij}\log(\beta^*_{ij}/\pi_{ij})$"),
}
# The unweighted self-divergence, drawn beside the terms (hatched) and not summed into F.
_CANONICAL_SELF_DIVERGENCE_LABELS = ("self-divergence $D(q_i\\|p_i)$, not in the sum",
                                     "self-divergence\n" r"$D(q_i\|p_i)$ (not summed)")
# Router columns of the canonical diagnostic, drawn beside the energy: (column, label).
_CANONICAL_ROUTER_SERIES: Tuple[Tuple[str, str], ...] = (
    ("canonical_beta_tv", r"$\frac{1}{2}\sum_j|\beta_{ij}-\beta^*_{ij}|$ (model vs canonical router)"),
    ("canonical_self_weight", r"canonical self weight $\beta^*_{ii}$"),
)
_CANONICAL_FE_NOTE = ("Canonical F at the beliefs each router read (diagnose_canonical_free_energy): a "
                      "tau-tempered diagnostic, not an ELBO. No step on this route optimizes it.")
# The batch the plotted canonical F was evaluated on: the val_* columns of _val_diagnostics (the validation split
# val_ce is scored on) or, for a run that logged only the log-step columns, the training batch (audit 2026-10-16
# F83): (title prefix, note sentence).
_CANONICAL_FE_POPULATIONS: Dict[str, Tuple[str, str]] = {
    "held_out": ("Held-out", "F on the held-out diagnostic sequences, CE over the validation split."),
    "training_batch": ("Training-batch", "F on the training batch of each log step, CE over the validation "
                                         "split: the two come from different data."),
}


def _canonical_fe_population(population: str) -> Tuple[str, str]:
    r"""``(title prefix, note sentence)`` of a canonical-F population; ValueError for an unknown one."""
    if population not in _CANONICAL_FE_POPULATIONS:
        raise ValueError(f"canonical free-energy population must be one of "
                         f"{sorted(_CANONICAL_FE_POPULATIONS)}, got {population!r}")
    return _CANONICAL_FE_POPULATIONS[population]


def _canonical_router_panel(ax, history: Dict, step: np.ndarray) -> bool:
    r"""Draw the canonical-router columns of ``history`` over ``step``; False when none is finite."""
    drawn = False
    for index, (key, label) in enumerate(_CANONICAL_ROUTER_SERIES):
        if key not in history:
            continue
        values = _np(history[key]).astype(float)
        if not np.isfinite(values).any():
            continue
        ax.plot(step, values, "o-", ms=2, lw=1.1, color=_CB[(index + 3) % len(_CB)], label=label)
        drawn = True
    if drawn:
        ax.set(xlabel="training step", ylabel="fraction of row mass", ylim=(0.0, 1.0),
               title="Model router against the canonical router")
        ax.legend(frameon=False, loc="best")
        if _set_finite_xlim(ax, step):
            _step_xaxis(ax)
    return drawn


@register_figure("canonical_free_energy_relationship")
def plot_canonical_free_energy_relationship(
    history: Dict,                       # step, self_coupling, belief_coupling, attention_entropy, val_ce
    #                                      (+ canonical_beta_tv, canonical_self_weight when logged)

    *,
    lambda_beta: 'float | np.ndarray' = 1.0,
    population:  str           = "held_out",     # "held_out" | "training_batch" (_CANONICAL_FE_POPULATIONS)
    path:        Optional[str] = None,
):
    r"""The canonical free energy at the captured beliefs beside the held-out CE (a diagnostic).

    Left: the per-token canonical :math:`F` of ``diagnose_canonical_free_energy`` (the self-coupling
    :math:`\alpha_iD(q_i\|p_i)+R(\alpha_i)` plus the :math:`\lambda_\beta`-weighted belief coupling and router
    entropy of the canonical router :math:`\beta^*`) and the validation CE on twin axes, each a faint raw
    line under a rolling mean, with their Pearson association in the title. Right, when logged: the total
    variation between the model's router and :math:`\beta^*` (and the canonical self weight). On the
    projected routes nothing optimizes :math:`F`, so the figure makes no E-step or co-descent claim.
    ``population`` names the batch the :math:`F` columns were evaluated on, in the title and the note: the
    held-out diagnostic sequences (the ``val_*`` columns) or the training batch of each log step, which
    pairs the correlation across two populations.
    """
    title_prefix, population_note = _canonical_fe_population(population)
    step, _comps, total, ce = _fe_terms(history, lambda_beta)
    keep = np.isfinite(total) & np.isfinite(ce)
    router_history = {key: _np(history[key]).astype(float)[keep]
                      for key, _ in _CANONICAL_ROUTER_SERIES if key in history}
    step, total, ce = step[keep], total[keep], ce[keep]
    has_router = any(np.isfinite(values).any() for values in router_history.values())
    w = max(5, total.size // 80)
    with _compact_style():
        fig, axes = plt.subplots(1, 2 if has_router else 1, squeeze=False,
                                 figsize=(MAX_FIGURE_WIDTH_IN if has_router else 4.6, 3.0))
        ax = axes[0][0]
        ax.plot(step, total, lw=0.7, color=_CB[0], alpha=0.25)
        flab = f"canonical F (left) = {total[-1]:.2f}" if step.size else "canonical F (left)"
        ln1, = ax.plot(step, _rolling_mean(total, w), lw=1.6, color=_CB[0], label=flab)
        ax.set(xlabel="training step", ylabel="canonical F (nats/token)")
        ax.yaxis.label.set_color(_CB[0]); ax.tick_params(axis="y", colors=_CB[0])
        if step.size and _set_finite_xlim(ax, step):
            _step_xaxis(ax)
        ax2 = ax.twinx()
        ax2.spines["right"].set_visible(True)
        ax2.plot(step, ce, lw=0.7, color=_CB[1], alpha=0.25)
        clab = f"val CE (right) = {ce[-1]:.2f}" if step.size else "val CE (right)"
        ln2, = ax2.plot(step, _rolling_mean(ce, w), lw=1.6, color=_CB[1], ls="--", label=clab)
        ax2.set_ylabel("validation CE (nats/token)", color=_CB[1]); ax2.tick_params(axis="y", colors=_CB[1])
        ax2.grid(False)
        title = f"{title_prefix} canonical F (diagnostic) and validation CE"
        if total.size > 2:
            title += rf" (Pearson $r={float(np.corrcoef(total, ce)[0, 1]):+.2f}$)"
        ax.set_title(title)
        ax.legend([ln1, ln2], [ln1.get_label(), ln2.get_label()], loc="upper right", frameon=False)
        if has_router:
            _canonical_router_panel(axes[0][1], router_history, step)
        fig.text(0.01, 0.005, f"{_CANONICAL_FE_NOTE} {population_note}", fontsize=6, color="#444444",
                 ha="left", va="bottom")
        fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
        return _save(fig, path)


@register_figure("canonical_free_energy_decomposition")
def plot_canonical_free_energy_decomposition(
    history: Dict,                       # step, self_coupling, belief_coupling, attention_entropy
    #                                      (+ self_divergence, canonical_beta_tv, canonical_self_weight)

    *,
    lambda_beta: 'float | np.ndarray' = 1.0,
    population:  str           = "held_out",     # "held_out" | "training_batch" (_CANONICAL_FE_POPULATIONS)
    path:        Optional[str] = None,
):
    r"""Terms of the canonical free energy at the captured beliefs, at the last evaluation and by training third.

    Panel A: the terms of the per-token canonical :math:`F` at the last logged evaluation, the self-coupling
    :math:`\alpha_iD(q_i\|p_i)+R(\alpha_i)` and the :math:`\lambda_\beta`-weighted belief coupling and router
    entropy of :math:`\beta^*`, which sum to :math:`F`, and, hatched and not summed, the unweighted
    self-divergence :math:`D(q_i\|p_i)`, so the regularizer :math:`R(\alpha_i)` is not read as divergence.
    Panel B: the same terms as medians over the early, mid and late thirds of training. Panel C, when
    logged: the total variation between the model's router and :math:`\beta^*` (and the canonical self
    weight). The axes are logarithmic when every drawn value is positive. A diagnostic: nothing on the
    projected routes optimizes :math:`F`. ``population`` names, in the note, the batch the terms were evaluated
    on (the held-out diagnostic sequences or the training batch of each log step).
    """
    _title_prefix, population_note = _canonical_fe_population(population)
    step, comps, _total, _ce = _fe_terms(history, lambda_beta)
    names = [_CANONICAL_FE_LABELS[k][1] for k, _ in comps]
    labels = [_CANONICAL_FE_LABELS[k][0] for k, _ in comps]
    series = [c for _, c in comps]
    hatches = [None] * len(series)
    if "self_divergence" in history and np.isfinite(_np(history["self_divergence"]).astype(float)).any():
        series.append(_np(history["self_divergence"]).astype(float))
        labels.append(_CANONICAL_SELF_DIVERGENCE_LABELS[0])
        names.append(_CANONICAL_SELF_DIVERGENCE_LABELS[1])
        hatches.append("///")
    colors = [_CB[0], _CB[1], _CB[2], _CB[0]][:len(series)]
    has_router = any(key in history and np.isfinite(_np(history[key]).astype(float)).any()
                     for key, _ in _CANONICAL_ROUTER_SERIES)
    last = np.array([s[-1] for s in series], dtype=float)
    thirds = np.array_split(np.arange(step.size), 3)
    medians = [[float(np.median(s[idx])) if idx.size else np.nan for idx in thirds] for s in series]
    with _compact_style():
        n_panels = 3 if has_router else 2
        fig, axes = plt.subplots(1, n_panels, squeeze=False, figsize=(MAX_FIGURE_WIDTH_IN, 3.0))
        axes = axes[0]
        y = np.arange(len(series))[::-1]
        for yi, value, color, hatch in zip(y, last, colors, hatches):
            axes[0].barh(yi, value, color="white" if hatch else color, edgecolor=color, hatch=hatch,
                         height=0.62, lw=0.8)
            axes[0].annotate(f"{value:.2f}", xy=(value, yi), xytext=(3, 0), textcoords="offset points",
                             va="center", ha="left")
        positive = bool(np.all(last[np.isfinite(last)] > 0))
        if positive:
            axes[0].set_xscale("log")
            axes[0].set_xlim(float(np.nanmin(last)) * 0.5, float(np.nanmax(last)) * 4.0)
        axes[0].set_yticks(y); axes[0].set_yticklabels(names)
        axes[0].set(xlabel="nats/token" + (" (log scale)" if positive else ""),
                    title=f"Terms of F at step {int(step[-1]):,}" if step.size else "Terms of F")
        centers = np.arange(3)
        width = min(0.2, 0.8 / max(len(series), 1))
        off0 = (len(series) - 1) / 2.0
        for j, (meds, color, hatch) in enumerate(zip(medians, colors, hatches)):
            axes[1].bar(centers + (j - off0) * width, meds, width=width, color="white" if hatch else color,
                        edgecolor=color, hatch=hatch, lw=0.8, label=labels[j])
        if positive and np.all(np.asarray(medians, dtype=float)[np.isfinite(medians)] > 0):
            axes[1].set_yscale("log")
        axes[1].set_xticks(centers); axes[1].set_xticklabels(["early", "mid", "late"])
        axes[1].set(xlabel="training third", ylabel="median (nats/token)", title="Terms by training third")
        axes[1].legend(frameon=False, loc="best")
        if has_router:
            _canonical_router_panel(axes[2], history, step)
        fig.text(0.01, 0.005, f"{_CANONICAL_FE_NOTE} {population_note}", fontsize=6, color="#444444",
                 ha="left", va="bottom")
        fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
        return _save(fig, path)


def _grad_norm_decomp_fig(
    history: Dict,
    spec:    list,                       # [(column_key, latex_label, color, role_name), ...]
    title:   str,                        # base title; a "(role / role)" suffix is appended from the
    #                                      components ACTUALLY plotted, so it never promises a role the
    #                                      figure does not show (e.g. phi when e_phi_lr=0)
    ylabel:  str,
    path:    Optional[str],

    *,
    spikes:  Optional[Mapping[str, object]] = None,   # {step, key: [...]} of divergence-spike rows
):
    r"""Shared body for the mu / sigma / phi gradient-norm decomposition figures (M-step and E-step).

    Each series is a faint raw line under a rolling-mean trend on a log y (norms are strictly positive
    and span orders of magnitude), with its final value folded into the legend label (a separate
    right-edge tag collides with the upper-right legend). A component that is identically zero / absent
    (a dead or off substep) drops out under the log-y positive mask rather than flatlining at 0, and its
    role is omitted from the title suffix.
    """
    step = _np(history["step"]).astype(float)
    with _compact_style():
        fig, ax = plt.subplots(figsize=(5.4, 3.2))
        w = max(5, step.size // 80)
        roles = []
        for key, label, color, role in spec:
            if key not in history:
                continue
            v = _np(history[key]).astype(float)
            keep = np.isfinite(v) & (v > 0)                       # log y: drop non-positive (dead/off-substep) points
            x, vv = step[keep], v[keep]
            if not vv.size:
                continue
            roles.append(role)
            ax.plot(x, vv, lw=0.6, color=color, alpha=0.25)
            ax.plot(x, _rolling_mean(vv, w), lw=1.5, color=color, label=f"{label} (last {vv[-1]:.3g})")
        if roles:
            ax.set_yscale("log")
            marked = sum(_overlay_spike_markers(ax, spikes, key, color, logy=True)
                         for key, _, color, _ in spec if key in history)
            if step.size and _set_finite_xlim(ax, step):
                _step_xaxis(ax)
            ax.legend(fontsize=6.5, frameon=False, loc="best")
            title = f"{title} ({' / '.join(roles)})"
            if marked:
                title += "\nx: divergence-spike rows"
        ax.set(xlabel="training step", ylabel=ylabel, title=title)
        fig.tight_layout()
        return _save(fig, path)


# Named parameter groups of the per-group optimizer columns (implementation contract 2026-09-28):
# ``grad_norm_group_<g>``, ``param_norm_group_<g>`` and ``update_ratio_group_<g>`` on log rows,
# assigned by parameter NAME, so the per-type tables and the context-processing maps are separate.
# "sigma_log_embed" also holds the block_cholesky table sigma_chol_embed, and "per_type_table" the
# equal-size capacity-control table (vfe3.train.PARAMETER_GROUPS).
PARAMETER_GROUPS = ("mu_embed", "sigma_log_embed", "phi_embed", "pos_phi", "decode",
                    "relational_score", "relational_content", "block_mlp", "norm", "per_type_table", "other")
_TABLE_GROUPS = ("mu_embed", "sigma_log_embed", "phi_embed", "pos_phi", "decode", "per_type_table")
_CONTEXT_GROUPS = ("relational_score", "relational_content", "block_mlp", "norm", "other")
_GROUP_LABELS = {
    "mu_embed":           r"$\mu$ table",
    "sigma_log_embed":    r"$\Sigma$ table ($\log\sigma^2$ or Cholesky)",
    "phi_embed":          r"$\phi$ table",
    "pos_phi":            "positional frames",
    "decode":             r"decode ($W_{\mathrm{out}}$, $b_{\mathrm{out}}$)",
    "relational_score":   r"score maps $S_Q,S_K$",
    "relational_content": r"content maps $T,O$",
    "block_mlp":          "block MLP",
    "norm":               "normalization",
    "per_type_table":     "per-type capacity table",
    "other":              "other",
}
_GROUP_COLORS = {
    "mu_embed": _CB[0], "sigma_log_embed": _CB[1], "phi_embed": _CB[2], "pos_phi": _CB[4],
    "decode": _CB[3], "relational_score": _CB[0], "relational_content": _CB[1],
    "block_mlp": _CB[2], "norm": _CB[3], "per_type_table": _CB[5], "other": _CB[7],
}
# The legacy per-ROLE columns (grad_norm_mu / _sigma / _phi, weight_norm_*) aggregate optimizer groups
# by their build_optimizer role tag. The 'mu' role is everything outside the Sigma and phi tables.
_ROLE_LABELS = {
    "mu":    r"'mu' role: $\mu$ table, decode, MLPs, relational maps, norm",
    "sigma": r"'sigma' role: $\log\sigma^2$ table(s)",
    "phi":   r"'phi' role: $\phi$ table, positional frames",
}
_ROLE_NOTE = ("'mu' role = mu table, decode (W_out, b_out), block MLPs, relational maps, normalization; "
              "'phi' role = phi table, positional frames; 'sigma' role = log-variance table(s)")
_PULLBACK_COLUMNS = (
    "phi_ridge_direction_cosine_mean",
    "phi_pullback_damped_gen_cond_median", "phi_pullback_damped_gen_cond_max",
    "phi_group_trust_scale_mean", "phi_group_trust_scale_min",
    "phi_group_active_rows", "phi_group_chart_norm_max",
    # factor-scale halvings of the accepted step on both retractions (audit 2026-10-16 F14)
    "phi_group_backtracks_mean", "phi_group_backtracks_max",
)
# History columns each training-time dashboard reads; run_artifacts._save_figures selects these.
GRAD_NORM_DECOMPOSITION_COLUMNS = (
    tuple(f"grad_norm_group_{group}" for group in PARAMETER_GROUPS)
    + ("grad_norm_mu", "grad_norm_sigma", "grad_norm_phi"))
OPTIMIZER_GEOMETRY_COLUMNS = (
    _PULLBACK_COLUMNS
    + tuple(f"param_norm_group_{group}" for group in PARAMETER_GROUPS)
    + tuple(f"update_ratio_group_{group}" for group in PARAMETER_GROUPS)
    + ("sigma_log_mean", "sigma_log_sd",
       "weight_norm_mu", "weight_norm_sigma", "weight_norm_phi",
       "grad_norm_mu", "grad_norm_sigma", "grad_norm_phi"))


def _live(history: Mapping[str, object], key: str) -> bool:
    r"""Whether ``history[key]`` exists and has at least one finite sample."""
    return key in history and bool(np.isfinite(_np(history[key]).astype(float)).any())


# Line-end labels are read beside the data, so the decode label drops its parenthetical.
_GROUP_SHORT_LABELS = {**_GROUP_LABELS, "decode": "decode"}
# The two parameter families of the small-multiple figures, each in one color.
_FAMILIES = (("Per-type tables", _TABLE_GROUPS, _CB[0]),
             ("Context processing", _CONTEXT_GROUPS, _CB[2]))


def _group_panels(
    prefix:  str,
    ylabel:  str,

    *,
    exclude: Sequence[str] = (),
    short:   bool          = False,      # line-end labels (_GROUP_SHORT_LABELS)
) -> List[Dict[str, object]]:
    r"""Two log-y panels over the named groups of ``prefix``: per-type tables and context processing."""
    names = _GROUP_SHORT_LABELS if short else _GROUP_LABELS
    panels = []
    for title, groups in (("per-type tables", _TABLE_GROUPS),
                          ("context processing", _CONTEXT_GROUPS)):
        series = [(f"{prefix}{group}", names[group], _GROUP_COLORS[group])
                  for group in groups if group not in exclude]
        panels.append({"title": title, "ylabel": ylabel, "logy": True, "series": series})
    return panels


def _family_small_multiples(
    history:  Mapping[str, object],
    prefix:   str,                       # column prefix, e.g. "grad_norm_group_"
    ylabel:   str,
    suptitle: str,
    path:     Optional[str],

    *,
    max_cols: int = 5,
    spikes:   Optional[Mapping[str, object]] = None,   # {step, key: [...]} of divergence-spike rows
) -> object:
    r"""One small log-y panel per live parameter group of ``prefix``, grouped by family, no legends.

    Each family of :data:`_FAMILIES` starts a new row under a bold header, in its own color; a panel
    is titled with its group and draws the raw series faintly under a rolling geometric-mean trend
    (:func:`_log_rolling_mean`). Every panel shares one log y axis, so magnitudes compare across
    groups and families, and one step axis. With one panel per series nothing can overlap, which is
    why this layout replaces the overlaid two-panel form whose legends covered half of each panel.
    """
    step = _np(history["step"]).astype(float)
    families = [(title, [g for g in groups if _live(history, f"{prefix}{g}")], color)
                for title, groups, color in _FAMILIES]
    families = [family for family in families if family[1]]
    if not families:
        raise FigureSkipped(f"no finite {prefix}<group> column")
    ncols = max(1, min(int(max_cols), max(len(groups) for _, groups, _ in families)))
    rows = [(title if start == 0 else None, groups[start:start + ncols], color)
            for title, groups, color in families for start in range(0, len(groups), ncols)]
    with _compact_style():
        fig, axes = plt.subplots(len(rows), ncols, squeeze=False, sharex=True, sharey=True,
                                 figsize=(MAX_FIGURE_WIDTH_IN, 1.6 * len(rows) + 0.5))
        title_pt = float(plt.rcParams["axes.titlesize"])
        marked = 0
        for r, (header, groups, color) in enumerate(rows):
            for c in range(ncols):
                ax = axes[r][c]
                if c >= len(groups):
                    ax.axis("off")
                    continue
                v = _np(history[f"{prefix}{groups[c]}"]).astype(float)
                keep = np.isfinite(v) & (v > 0) & np.isfinite(step)
                x, vv = step[keep], v[keep]
                if vv.size > 50:
                    ax.plot(x, vv, lw=0.5, color=color, alpha=0.3)
                    ax.plot(x, _log_rolling_mean(vv, max(5, vv.size // 80)), lw=1.2, color=color)
                else:
                    ax.plot(x, vv, "o-", ms=2.0, lw=1.0, color=color)
                ax.set_yscale("log")
                marked += _overlay_spike_markers(ax, spikes, f"{prefix}{groups[c]}", color, logy=True)
                ax.set_title(_GROUP_LABELS[groups[c]])
                ax.tick_params(labelbottom=True, labelleft=c == 0)
                if c == 0:
                    ax.set_ylabel(ylabel)
                    if header is not None:
                        # Above the panel title: titlepad plus a mathtext-tall title line.
                        ax.annotate(header, xy=(0.0, 1.0), xycoords="axes fraction",
                                    xytext=(0.0, float(plt.rcParams["axes.titlepad"]) + 1.6 * title_pt),
                                    textcoords="offset points", ha="left", va="bottom",
                                    fontsize=title_pt, fontweight="bold")
        first = axes[0][0]
        if _set_finite_xlim(first, step):
            first.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True, steps=[1, 2, 2.5, 5, 10]))
            first.xaxis.set_major_formatter(FuncFormatter(_kstep))
        _tidy_log_axis(first.yaxis)                                  # final tick labels before layout
        fig.suptitle(f"{suptitle} (x: divergence-spike rows)" if marked else suptitle)
        fig.supxlabel("training step", fontsize=float(plt.rcParams["axes.labelsize"]))
        fig.tight_layout()
        return _save(fig, path)


@register_figure("grad_norm_decomposition")
def plot_grad_norm_decomposition(
    history: Dict,                       # step + grad_norm_group_<g> and/or legacy grad_norm_mu/_sigma/_phi

    *,
    spikes:  Optional[Mapping[str, object]] = None,   # the same columns over divergence-spike rows
    path:    Optional[str] = None,
):
    r"""M-STEP pre-clip parameter-gradient L2 norms over training, by named parameter group.

    With the ``grad_norm_group_<g>`` columns (``g`` in :data:`PARAMETER_GROUPS`) each live group gets
    its own panel (:func:`_family_small_multiples`): the per-type tables (mu, log-variance and phi
    tables, positional frames, decode) in one row block and the context-processing parameters (score
    maps, content maps, block MLP, normalization, other) in the next, all on one shared log y axis.
    A run that logged only the legacy per-ROLE columns falls back to those, labeled with what each
    role aggregates: the 'mu' role holds every parameter outside the Sigma and phi tables (decode,
    MLPs, relational maps, normalization), so it is not a mu-table gradient. Captured after
    ``unscale_`` and before clipping in ``train_step``; the E-step analogue is
    ``estep_grad_norm_decomposition``. Divergence-spike rows (``spikes``) are crosses, not line points.
    """
    if any(_live(history, f"grad_norm_group_{group}") for group in PARAMETER_GROUPS):
        return _family_small_multiples(
            history, "grad_norm_group_", r"pre-clip $\|\nabla L\|_2$",
            "M-step gradient norm by parameter group", path, spikes=spikes)
    spec = [
        ("grad_norm_mu",    _ROLE_LABELS["mu"],    _CB[0], "mu"),
        ("grad_norm_sigma", _ROLE_LABELS["sigma"], _CB[1], "sigma"),
        ("grad_norm_phi",   _ROLE_LABELS["phi"],   _CB[2], "phi"),
    ]
    return _grad_norm_decomp_fig(
        history, spec, "M-step gradient norm by optimizer role",
        r"pre-clip $\|\nabla_\theta L\|_2$ (per role)", path, spikes=spikes)


@register_figure("estep_grad_norm_decomposition")
def plot_estep_grad_norm_decomposition(
    history: Dict,                       # step + single-batch or microbatch-mean E-step grad norms

    *,
    path:    Optional[str] = None,
):
    r"""E-STEP per-component belief-gradient L2 norms over training: the mu / sigma / phi decomposition
    of the inference free-energy gradient.

    The companion to the M-step grad_norm_decomposition: where that shows how hard the LEARNING (M-step,
    parameter) gradient pushes each belief-component family, this shows how hard the INFERENCE (E-step)
    gradient ``\nabla F`` pushes the belief tuple ``(mu, Sigma, phi)`` itself -- the RAW gradient inside
    the last E-step iteration, before the Fisher / natural-gradient preconditioning, captured by
    model.forward(estep_grad_out=...). Accumulated steps use the arithmetic mean across contributing
    microbatches. A component reads 0 (dropped on the log y) when its substep is off (e.g. phi when
    e_phi_lr=0).
    """
    mean_keys = (
        "estep_grad_norm_mu_microbatch_mean",
        "estep_grad_norm_sigma_microbatch_mean",
        "estep_grad_norm_phi_microbatch_mean",
    )
    keys = (mean_keys if any(key in history for key in mean_keys)
            else ("estep_grad_norm_mu", "estep_grad_norm_sigma", "estep_grad_norm_phi"))
    spec = [
        (keys[0], r"$\|\nabla_\mu F\|_2$",    _CB[0], "mu"),
        (keys[1], r"$\|\nabla_\Sigma F\|_2$", _CB[1], "sigma"),
        (keys[2], r"$\|\nabla_\phi F\|_2$",   _CB[2], "phi"),
    ]
    return _grad_norm_decomp_fig(
        history, spec, "E-step gradient norm decomposition",
        r"E-step $\|\nabla F\|_2$ (per component)", path)


@register_figure("estep_convergence")
@register_figure("iterate_trajectory")
def plot_iterate_trajectory(
    trace:    Dict,                      # e_step_belief_trace output: mu, sigma, phi

    *,
    diagonal:          Optional[bool] = None,
    divergence_family: str = "renyi",
    evidence:           Optional[Mapping[str, object]] = None,
    path:               Optional[str] = None,
):
    r"""Belief motion across the configured E-step iterate trajectory.

    The projected routes compute routing and readout in distinct conditional stages and define no shared free
    energy, so panel A states that and the trace carries no free-energy series (audit 2026-10-16 F92). Panel B
    is the per-iteration step length of the captured states, shown without treating small final movement as
    convergence unless structured evidence establishes it.
    """
    from vfe3 import metrics
    objective_label, objective_units = _registered_divergence_axis(divergence_family)
    evidence = dict(evidence or {})
    if diagonal is None:
        diagonal = trace.get("diagonal_covariance")
    res = metrics.estep_residuals(trace["mu"], trace["sigma"], trace["phi"], diagonal=diagonal)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].text(0.5, 0.5,
                 "Shared free energy unavailable\nRouting and readout use distinct stages.\n"
                 "The captured state movement is shown at right.",
                 ha="center", va="center", transform=axes[0].transAxes, fontsize=9)
    axes[0].set_axis_off()
    trajectory_title = "E-step iterate trajectory"
    if evidence.get("fixed_point_evidence"):
        trajectory_title += " (fixed point established)"
    elif evidence.get("descent_evidence"):
        trajectory_title += " (descent established)"
    axes[0].set(xlabel="E-step inner iteration",
                ylabel=_objective_axis_text(objective_label, objective_units),
                title=trajectory_title)
    t = np.arange(1, _np(res["r_mu"]).shape[0] + 1)
    _median_band(axes[1], t, _np(res["r_mu"]), _CB[0], r"$r_\mu$")
    _median_band(axes[1], t, _np(res["r_sigma"]), _CB[1], r"$r_\Sigma$ (SPD)")
    _median_band(axes[1], t, _np(res["r_phi"]), _CB[2], r"$r_\phi$")
    if any(np.any(_np(res[key]) > 0) for key in ("r_mu", "r_sigma", "r_phi")):
        axes[1].set_yscale("log")
    axes[1].set(xlabel="E-step inner iteration", ylabel="belief step length",
                title="Final-iterate movement")
    axes[1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, path)


plot_estep_convergence = plot_iterate_trajectory


def _unavailable_probe_figure(record: Mapping[str, object], path: Optional[str]) -> object:
    """Render the recorded reason instead of an empty chart or an incorrect missing-data claim."""
    import textwrap
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.text(0.5, 0.55, "Unavailable for this configuration", ha="center", fontsize=11,
            transform=ax.transAxes)
    ax.text(0.5, 0.35, textwrap.fill(str(record.get("reason", "probe unavailable")), 66),
            ha="center", va="center", fontsize=9, transform=ax.transAxes)
    ax.set_axis_off()
    return _save(fig, path)


@register_figure("estep_character")
def plot_estep_character(
    record: Mapping[str, object],

    *,
    path:   Optional[str] = None,
):
    r"""Plot how far the belief E-step moves the belief with depth.

    The relative belief displacement per depth, with the direction cosine against the first step
    in the title. A flat displacement curve is one aggregation, not convergent inference.

    Raises :class:`FigureSkipped` when the probe is unavailable or covers only the single configured
    readout (``depth_sweep_available`` false): there are no E-step iterates, and the one point is
    the layer-0 message-to-prior ratio, which ``diagnostics/residual_ratio.csv`` reports per layer and
    per token.
    """
    if record.get("available") is False:
        raise FigureSkipped(str(record.get("reason", "E-step character probe unavailable")))
    if record.get("depth_sweep_available") is False:
        raise FigureSkipped(
            "single configured readout: no E-step iterates; the layer-0 message-to-prior ratio is "
            "reported per layer by diagnostics/residual_ratio.csv")
    points = [p for p in record.get("points", []) if p.get("rel_displacement") is not None]
    fig, ax_move = plt.subplots(figsize=(5.6, 3.6))
    if not points:
        ax_move.text(0.5, 0.5, "no belief displacement recorded", ha="center", va="center",
                     transform=ax_move.transAxes, fontsize=9)
        ax_move.set_axis_off()
        return _save(fig, path)
    depths = np.asarray([p["belief_depth"] for p in points], dtype=float)
    displacement = np.asarray([p["rel_displacement"] for p in points], dtype=float)
    ax_move.plot(depths, displacement, "o-", color=_CB[0], label=r"$\|\Delta\mu\|/\|\mu_p\|$")
    trained_depth = float(record.get("trained_depth", np.nan))
    if np.isfinite(trained_depth):
        ax_move.axvline(trained_depth, color="#555555", ls=":", lw=1.0, label="trained depth")
    cosines = [p.get("cos_dir_vs_step1") for p in points if p.get("cos_dir_vs_step1") is not None]
    if cosines:
        ax_move.set_title(f"E-step character  (cos(dir, step 1) = {cosines[-1]:+.3f})")
    else:
        ax_move.set_title("E-step character")
    ax_move.set(xlabel="belief E-step iterations", ylabel="relative belief displacement")
    ax_move.legend(frameon=False, fontsize=8, loc="center right")
    fig.tight_layout()
    return _save(fig, path)


# Bar labels of plot_beta_channel_decomposition. Under a rope_mode other than 'none' the routing cost carries
# RoPE position as well as content (the writer flags energy_includes_rope_position), so content_channel is
# relabelled and the rotation-only arm is drawn beside it (audit 2026-10-16, F25).
BETA_CHANNEL_POSITIONAL_LABEL   = "positional prior"
BETA_CHANNEL_CONTENT_LABEL      = "content energy"
BETA_CHANNEL_ROUTING_COST_LABEL = "routing cost\n(content + RoPE position)"
BETA_CHANNEL_ROPE_LABEL         = "RoPE position\n(rotation only)"


@register_figure("beta_channel_decomposition")
def plot_beta_channel_decomposition(
    record: Mapping[str, object],

    *,
    path:   Optional[str] = None,
):
    r"""Plot what each channel of the attention weight is worth, in nats of cross-entropy.

    Each bar is :math:`\Delta\mathrm{CE} = \mathrm{CE}(\text{arm}) - \mathrm{CE}(\text{full})` of one separate
    ablation of :func:`vfe3.run_artifacts.collect_beta_channel_decomposition`. When the record flags
    ``energy_includes_rope_position``, the ``content_channel`` bar (the whole routing cost zeroed) is labelled
    as content plus RoPE position and the ``rope_position_channel`` arm (rotation :math:`R_i = I` inside the
    cost) is drawn beside it. The arms are not additive, so no difference of bars is shown as "content".
    A record without the flag (``rope_mode='none'``, or written before the flag existed) keeps two bars.
    """
    if record.get("available") is False:
        return _unavailable_probe_figure(record, path)
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    deltas = record.get("delta_ce") or {}
    rope_in_cost = bool(record.get("energy_includes_rope_position"))
    channels = [("positional_channel", BETA_CHANNEL_POSITIONAL_LABEL, _CB[1]),
                ("content_channel",
                 BETA_CHANNEL_ROUTING_COST_LABEL if rope_in_cost else BETA_CHANNEL_CONTENT_LABEL, _CB[0])]
    if rope_in_cost:
        channels.append(("rope_position_channel", BETA_CHANNEL_ROPE_LABEL, _CB[2]))
    labels, values, colors = [], [], []
    for key, label, color in channels:
        if deltas.get(key) is not None:
            labels.append(label)
            values.append(float(deltas[key]))
            colors.append(color)
    if not values:
        ax.text(0.5, 0.5, "no live beta softmax intercepted", ha="center", va="center",
                transform=ax.transAxes, fontsize=9)
        ax.set_axis_off()
        return _save(fig, path)
    bars = ax.bar(labels, values, color=colors, width=0.55)
    for bar, value in zip(bars, values):
        ax.annotate(f"{value:+.3f}", (bar.get_x() + bar.get_width() / 2.0, value),
                    ha="center", va="bottom", fontsize=8)
    ax.axhline(0.0, color="#555555", lw=0.8)
    ax.set(ylabel=r"$\Delta$CE when ablated (nats)",
           title="What each attention channel carries")
    if rope_in_cost and len(values) == 3:
        ax.set_xlabel("separate ablations; the bars do not add", fontsize=7, color="#555555")
    fig.tight_layout()
    return _save(fig, path)


@register_figure("context_sensitivity")
def plot_context_sensitivity(
    record: Mapping[str, object],

    *,
    path:   Optional[str] = None,
):
    r"""Plot where prefix context enters the belief, stage by stage.

    The ``raw_prior`` control reads 0 for a pure per-token lookup; each later bar is cumulative, so
    the increment is the context the belief E-step injects.
    """
    if record.get("available") is False:
        return _unavailable_probe_figure(record, path)
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    stages = record.get("stages") or {}
    labels = [name for name, value in stages.items() if value is not None]
    values = [float(stages[name]) for name in labels]
    if not values:
        ax.text(0.5, 0.5, "context-sensitivity probe not run", ha="center", va="center",
                transform=ax.transAxes, fontsize=9)
        ax.set_axis_off()
        return _save(fig, path)
    bars = ax.bar([name.replace("_", " ") for name in labels], values, color=_CB[:len(values)],
                  width=0.55)
    for bar, value in zip(bars, values):
        ax.annotate(f"{value:.3f}", (bar.get_x() + bar.get_width() / 2.0, value),
                    ha="center", va="bottom", fontsize=8)
    ax.set(ylabel="relative belief shift (prefix randomized)",
           title="Where context enters the belief")
    fig.tight_layout()
    return _save(fig, path)


@register_figure("phi_numerics_reference")
def plot_phi_numerics_reference(
    record: Mapping[str, object],

    *,
    path:   Optional[str] = None,
):
    r"""Plot fp32/fp64 flat-closure references and the group-product factor radii."""
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    flatness = record.get("flatness", {})
    labels = ["holonomy", "cocycle", "inverse"]
    fp32 = [
        float(flatness.get("numerical_holonomy_fp32_rel", np.nan)),
        float(flatness.get("numerical_cocycle_fp32_rel", np.nan)),
        float(flatness.get("inverse_consistency_fp32_rel", np.nan)),
    ]
    fp64 = [
        float(flatness.get("numerical_holonomy_fp64_rel", np.nan)),
        float(flatness.get("numerical_cocycle_fp64_rel", np.nan)),
        float(flatness.get("inverse_consistency_fp64_rel", np.nan)),
    ]
    x = np.arange(len(labels))
    axes[0].bar(x - 0.18, fp32, 0.36, color=_CB[0], label="fp32")
    axes[0].bar(x + 0.18, fp64, 0.36, color=_CB[1], label="fp64 reference")
    axes[0].set_xticks(x, labels)
    finite = [value for value in fp32 + fp64 if np.isfinite(value) and value > 0.0]
    if finite:
        axes[0].set_yscale("log")
        # Printed values: on a log axis a bar's visible height is set by the arbitrary axis floor.
        for offset, values in ((-0.18, fp32), (0.18, fp64)):
            for xi, value in zip(x + offset, values):
                if np.isfinite(value) and value > 0.0:
                    axes[0].annotate(_plain_tick(value), (xi, value), xytext=(0, 2),
                                     textcoords="offset points", ha="center", va="bottom", fontsize=7)
        axes[0].set_ylim(min(finite) / 10.0, max(finite) * 1e3)
    else:
        axes[0].text(0.5, 0.92, "zero at displayed precision",
                     transform=axes[0].transAxes, ha="center", va="top", fontsize=8)
    axes[0].set(ylabel="relative numerical residual", title="Nominally flat closure")
    axes[0].legend(frameon=False, fontsize=8)

    right_chart = record.get("right_chart", {})
    if right_chart:
        left_chart = record.get("chart", {})
        factor_labels = ["left p95", "left max", "right p95", "right max"]
        factor_values = [
            float(left_chart.get("phi_matrix_norm_p95", np.nan)),
            float(left_chart.get("phi_matrix_norm_max", np.nan)),
            float(right_chart.get("phi_matrix_norm_p95", np.nan)),
            float(right_chart.get("phi_matrix_norm_max", np.nan)),
        ]
        axes[1].bar(np.arange(len(factor_labels)), factor_values, color=[_CB[0], _CB[0], _CB[3], _CB[3]])
        axes[1].set_xticks(np.arange(len(factor_labels)), factor_labels, rotation=20, ha="right")
        axes[1].set(ylabel="embedded Frobenius norm", title="Exact group-product factor radii")
    else:
        axes[1].set(title="Exact positional composition")
        axes[1].text(0.5, 0.5, "Positional composition inactive",
                     transform=axes[1].transAxes, ha="center", va="center")
    fig.tight_layout()
    return _save(fig, path)


# ---------------------------------------------------------------------------
# History dashboards: small-multiples over metrics.csv columns the training loop
# already logs but no standard figure surfaced (geometry / E-step / validation /
# optimizer geometry). Shared helpers below; one registered figure per family.
# ---------------------------------------------------------------------------

SPIKE_MARKER_NOTE = "x: divergence-spike rows, drawn as markers and left out of the lines"


def _overlay_spike_markers(
    ax:     object,
    spikes: Optional[Mapping[str, object]],   # {step, key: [...]} over the divergence-spike rows
    key:    str,
    color:  str,

    *,
    logy:   bool = False,
) -> int:
    r"""Draw the spike rows' ``key`` values as crosses on ``ax`` and widen the y range to show them.

    Spike rows are forced between the log and eval rows by a reported divergence spike
    (:mod:`vfe3.metrics_schema`); the readers leave them out of every line and rolling mean, and
    this overlays them as markers instead. Returns the number of markers drawn.
    """
    if not spikes or key not in spikes or "step" not in spikes:
        return 0
    x = _np(spikes["step"]).astype(float).reshape(-1)
    v = _np(spikes[key]).astype(float).reshape(-1)
    keep = np.isfinite(x) & np.isfinite(v)
    if logy:
        keep &= v > 0
    if not keep.any():
        return 0
    x, v = x[keep], v[keep]
    lo, hi = ax.get_ylim()
    ax.plot(x, v, ls="none", marker="x", ms=3.5, mew=0.9, color=color, zorder=6)
    vmin, vmax = float(v.min()), float(v.max())
    if logy:
        new_lo = min(lo, vmin / 1.3) if lo > 0 else vmin / 1.3
        new_hi = max(hi, vmax * 1.3)
    else:
        span = max(hi - lo, abs(vmax - vmin), 1e-12)
        new_lo = min(lo, vmin - 0.05 * span)
        new_hi = max(hi, vmax + 0.05 * span)
    ax.set_ylim(new_lo, new_hi)
    return int(v.size)


def _series_panel(
    ax,
    history: Dict,
    step:    np.ndarray,
    specs:   list,                       # [(column_key, label, color), ...]

    *,
    logy:    bool = False,
) -> int:
    r"""Plot each PRESENT, finite series of ``specs`` onto ``ax`` (a faint raw line under a rolling-mean
    trend for a long series, ``o-`` markers for a short one), masking non-finite samples per series so
    an eval-cadence column (NaN on non-eval rows) draws only on its eval steps. Returns the series count
    actually drawn.

    The rolling mean applies only above 50 finite samples (as in :func:`plot_trajectory`): an
    eval-cadence series of ten points is drawn point by point, not smoothed through."""
    drawn = 0
    for key, label, color in specs:
        if key not in history:
            continue
        v = _np(history[key]).astype(float)
        keep = np.isfinite(v)
        if logy:
            keep = keep & (v > 0)                                # a log axis drops non-positive samples
        x, vv = step[keep], v[keep]
        if not vv.size:
            continue
        drawn += 1
        if vv.size > 50:
            ax.plot(x, vv, lw=0.6, color=color, alpha=0.25)
            ax.plot(x, _rolling_mean(vv, max(5, vv.size // 80)), lw=1.5, color=color,
                    label=f"{label} (last {vv[-1]:.3g})")
        else:
            ax.plot(x, vv, "o-", ms=2.5, lw=1.2, color=color, label=f"{label} (last {vv[-1]:.3g})")
    if drawn:
        if logy:
            ax.set_yscale("log")
        ax.legend(fontsize=6, frameon=False, loc="best")
    return drawn


# Line-end labels of the direct-label dashboards (plot_optimizer_geometry).
_LABEL_PT = 6.0                          # label font size (= MIN_TEXT_PT)
_LABEL_GAP_PT = 1.0                      # clear space between two stacked labels
_LABEL_MIN_CONTRAST = 4.5                # WCAG minimum for small text on white
_LABEL_X_PT = 9.0                        # label left edge, right of the axes' right edge
_LEADER_X_PT = (1.5, 7.5)                # leader start and end, right of the line end
_LEADER_MIN_PT = 1.0                     # a label displaced by less than this gets no leader


def _pack_labels(
    y:   np.ndarray,                     # (n,) anchor positions, any order
    gap: float,                          # minimum spacing between consecutive labels
    lo:  float,                          # lowest allowed position
    hi:  float,                          # highest allowed position
) -> np.ndarray:
    r"""Least-squares label positions at least ``gap`` apart that keep the anchors' order.

    With the anchors sorted, :math:`y_{(1)}\le\dots\le y_{(n)}`, the positions solve
    :math:`\min_p\sum_i(p_i-y_{(i)})^2` subject to :math:`p_{i+1}-p_i\ge g` and :math:`lo\le p_i\le hi`.
    Substituting :math:`z_i=p_i-(i-1)g` turns the spacing constraint into :math:`z` nondecreasing,
    which pool-adjacent-violators solves exactly; the bounds become the box
    :math:`lo\le z_i\le hi-(n-1)g`, and clipping the isotonic solution onto it is optimal.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if n == 0:
        return y
    order = np.argsort(y, kind="stable")
    shift = gap * np.arange(n)
    sums: List[float] = []
    counts: List[int] = []
    for value in y[order] - shift:                               # pool adjacent violators
        sums.append(float(value))
        counts.append(1)
        while len(sums) > 1 and sums[-2] / counts[-2] > sums[-1] / counts[-1]:
            total, count = sums.pop(), counts.pop()
            sums[-1] += total
            counts[-1] += count
    fitted = np.concatenate([np.full(c, s / c) for s, c in zip(sums, counts)])
    fitted = np.clip(fitted, lo, max(lo, hi - gap * (n - 1)))
    out = np.empty(n)
    out[order] = fitted + shift
    return out


def _label_color(color: str) -> tuple:
    r"""The line color darkened toward black, hue kept, until its contrast on white
    :math:`1.05/(L+0.05)` reaches :data:`_LABEL_MIN_CONTRAST` (:math:`L` the WCAG relative
    luminance). Orange and pink lines are too faint as 6 pt text otherwise."""
    from matplotlib.colors import to_rgb

    def luminance(rgb: np.ndarray) -> float:
        linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        return float(0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2])

    rgb = np.array(to_rgb(color))
    for _ in range(60):
        if 1.05 / (luminance(rgb) + 0.05) >= _LABEL_MIN_CONTRAST:
            break
        rgb = rgb * 0.95
    return tuple(float(c) for c in rgb)


def _fit_ylim(ax: object, values: Sequence[np.ndarray], *, logy: bool, pad: float = 0.04) -> None:
    r"""Y range = the drawn values' range padded by ``pad`` of its span (in :math:`\log_{10}` on a log
    axis), so no decade above or below the data is left empty."""
    v = np.concatenate([np.asarray(a, dtype=float).ravel() for a in values])
    v = v[np.isfinite(v) & ((v > 0) if logy else True)]
    if not v.size:
        return
    lo, hi = float(v.min()), float(v.max())
    if logy:
        a, b = np.log10(lo), np.log10(hi)
        margin = pad * (b - a) if b > a else 0.05
        ax.set_ylim(10.0 ** (a - margin), 10.0 ** (b + margin))
    else:
        margin = pad * (hi - lo) if hi > lo else max(0.05 * abs(lo), 1e-12)
        ax.set_ylim(lo - margin, hi + margin)


def _labeled_series_panel(
    ax,
    history: Dict,
    step:    np.ndarray,
    specs:   list,                       # [(column_key, label, color), ...]

    *,
    logy:    bool = False,
) -> int:
    r"""Plot each present series like :func:`_series_panel`, but name the lines at their right ends.

    The trend of a long series is a rolling mean, taken in log space on a log axis
    (:func:`_log_rolling_mean`). The y range is fitted to the drawn values; with two or more series
    each line gets a label in a gutter right of the axes (:func:`_add_direct_labels`), so no legend
    covers the data. Returns the series count drawn.
    """
    entries, drawn_values = [], []
    for key, label, color in specs:
        if key not in history:
            continue
        v = _np(history[key]).astype(float)
        keep = np.isfinite(v) & np.isfinite(step)
        if logy:
            keep = keep & (v > 0)
        x, vv = step[keep], v[keep]
        if not vv.size:
            continue
        if vv.size > 50:
            window = max(5, vv.size // 80)
            trend = _log_rolling_mean(vv, window) if logy else _rolling_mean(vv, window)
            ax.plot(x, vv, lw=0.6, color=color, alpha=0.25)
            ax.plot(x, trend, lw=1.3, color=color)
            drawn_values += [vv, trend]
        else:
            trend = vv
            ax.plot(x, vv, "o-", ms=2.5, lw=1.2, color=color)
            drawn_values.append(vv)
        entries.append((label, color, float(x[-1]), float(trend[-1])))
    if not entries:
        return 0
    if logy:
        ax.set_yscale("log")
    _fit_ylim(ax, drawn_values, logy=logy)
    if logy:
        _tidy_log_axis(ax.yaxis)                                 # final tick labels before the layout
    if len(entries) > 1:
        _add_direct_labels(ax, entries)
    return len(entries)


def _add_direct_labels(
    ax:      object,
    entries: Sequence[tuple],            # (text, color, x_end, y_end), anchors in data coordinates
) -> None:
    r"""Create the line-end labels and leaders of ``ax``; :func:`_settle_direct_labels` places them.

    The texts are created right of the axes before the layout pass, so ``tight_layout`` reserves the
    gutter they need; only their heights change afterwards, never their widths.
    """
    from matplotlib.lines import Line2D

    labels = []
    for text, color, x_end, y_end in entries:
        label = ax.text(1.0, 0.5, text, transform=ax.transAxes, color=_label_color(color),
                        fontsize=_LABEL_PT, ha="left", va="center", clip_on=False)
        leader = Line2D([1.0, 1.0], [0.5, 0.5], transform=ax.transAxes, color=color, lw=0.6,
                        clip_on=False, visible=False)
        ax.add_line(leader)
        labels.append({"text": label, "leader": leader, "x": x_end, "y": y_end})
    ax._vfe3_direct_labels = labels
    _settle_axes_labels(ax)


def _settle_axes_labels(ax: object) -> None:
    r"""Place the direct labels of one axes for its current position (axes-fraction coordinates).

    Label spacing comes from the tallest measured label (a superscripted mathtext label is taller
    than the font size); :func:`_pack_labels` keeps the anchors' order and moves each label as little
    as it can, and a leader joins a displaced label to its line end.
    """
    labels = getattr(ax, "_vfe3_direct_labels", None)
    if not labels:
        return
    fig = ax.figure
    position = ax.get_position()
    width_pt = position.width * fig.get_figwidth() * 72.0
    height_pt = position.height * fig.get_figheight() * 72.0
    to_axes = ax.transAxes.inverted()
    anchors = np.array([to_axes.transform(ax.transData.transform((lab["x"], lab["y"])))
                        for lab in labels])                      # (n, 2) axes fraction
    renderer = fig.canvas.get_renderer()
    tallest_pt = max(lab["text"].get_window_extent(renderer).height for lab in labels) * 72.0 / fig.dpi
    gap = (tallest_pt + _LABEL_GAP_PT) / height_pt
    placed = _pack_labels(anchors[:, 1], gap, 0.5 * gap, 1.0 - 0.5 * gap)
    x_text = 1.0 + _LABEL_X_PT / width_pt
    for lab, (xa, ya), yl in zip(labels, anchors, placed):
        lab["text"].set_position((x_text, yl))
        displaced = abs(yl - ya) * height_pt > _LEADER_MIN_PT
        lab["leader"].set_visible(bool(displaced))
        if displaced:
            lab["leader"].set_data([xa + _LEADER_X_PT[0] / width_pt, 1.0 + _LEADER_X_PT[1] / width_pt],
                                   [ya, yl])


def _settle_direct_labels(fig: object) -> None:
    r""":func:`_settle_axes_labels` on every axes of ``fig`` (after the final layout pass)."""
    for ax in fig.axes:
        _settle_axes_labels(ax)


def _history_dashboard(
    history:  Dict,
    panels:   list,                      # [{title, ylabel, series:[(key,label,color)], [logy]}, ...]
    suptitle: str,
    path:     Optional[str],

    *,
    ncols:    int           = 3,
    note:     Optional[str] = None,      # one-line figure footnote (e.g. what a legend role holds)
    labels:   str           = "legend",  # "legend" (in-axes legend) | "direct" (line-end labels)
    spikes:   Optional[Mapping[str, object]] = None,   # {step, key: [...]} of divergence-spike rows
) -> object:
    r"""Small-multiples dashboard over training history: one mini-panel per group of commensurate
    columns. A panel whose every column is absent or all-NaN is dropped (so a run missing a metric
    family simply shows fewer panels), and the grid self-sizes to the surviving panel count: at most
    three columns, never more columns than live panels, and at most :data:`MAX_FIGURE_WIDTH_IN` wide
    with text of at least :data:`MIN_TEXT_PT`.

    A panel may carry its own ``"data"`` history (``{step, key: [...]}``) and ``"xlabel"``; it then
    plots against that step axis instead of the shared one (an eval-cadence long-format table next
    to log-cadence columns). ``labels="direct"`` names each line at its right end instead of drawing
    a legend (:func:`_labeled_series_panel`), for panels of several series that a legend would cover.
    ``spikes`` (the same columns over the divergence-spike rows) is overlaid as crosses on the panels
    that plot against the shared step axis (:func:`_overlay_spike_markers`)."""
    if labels not in ("legend", "direct"):
        raise ValueError(f"labels must be 'legend' or 'direct', got {labels!r}")
    live = []
    for p in panels:
        source = p.get("data", history)
        present = [(k, lab, c) for (k, lab, c) in p["series"]
                   if k in source and bool(np.isfinite(_np(source[k]).astype(float)).any())]
        if present:
            live.append({**p, "series": present})
    with _compact_style():
        if not live:
            fig, ax = plt.subplots(figsize=(3.6, 2.4))
            ax.text(0.5, 0.5, "no history columns present", ha="center", va="center")
            ax.axis("off")
            return _save(fig, path)
        ncols = max(1, min(int(ncols), 3, len(live)))
        nrows = (len(live) + ncols - 1) // ncols
        width = MAX_FIGURE_WIDTH_IN if ncols > 1 else MAX_FIGURE_WIDTH_IN / 2.0
        fig, axes = plt.subplots(nrows, ncols, figsize=(width, 2.3 * nrows + 0.35), squeeze=False)
        marked = 0
        for i, p in enumerate(live):
            ax = axes[i // ncols][i % ncols]
            source = p.get("data", history)
            panel_step = _np(source["step"]).astype(float)
            draw = _labeled_series_panel if labels == "direct" else _series_panel
            draw(ax, source, panel_step, p["series"], logy=bool(p.get("logy", False)))
            if "data" not in p:
                for key, _, color in p["series"]:
                    marked += _overlay_spike_markers(ax, spikes, key, color, logy=bool(p.get("logy", False)))
            ax.set(xlabel=p.get("xlabel", "training step"), ylabel=p["ylabel"], title=p["title"])
            if panel_step.size and _set_finite_xlim(ax, panel_step):
                _step_xaxis(ax)
        for j in range(len(live), nrows * ncols):                # blank the unused grid cells
            axes[j // ncols][j % ncols].axis("off")
        fig.suptitle(suptitle)
        # Direct labels sit in a gutter right of each panel: keep a clear gap before the next
        # column's y label.
        w_pad = 1.8 if labels == "direct" else None
        if marked:
            note = SPIKE_MARKER_NOTE if not note else f"{note}; {SPIKE_MARKER_NOTE}"
        if note:
            fig.text(0.01, 0.005, note, fontsize=6, color="#444444", ha="left", va="bottom",
                     wrap=True)
            fig.tight_layout(rect=(0.0, 0.035, 1.0, 1.0), w_pad=w_pad)
        else:
            fig.tight_layout(w_pad=w_pad)
        _settle_direct_labels(fig)
        return _save(fig, path)


def _layer_colors(n_layers: int) -> List[str]:
    r"""One sequential (viridis) color per layer, so depth reads as a gradient across curves."""
    from matplotlib.colors import to_hex
    return [to_hex(plt.cm.viridis(index / max(n_layers - 1, 1) * 0.9)) for index in range(n_layers)]


def _long_rows(
    table:   Mapping[str, object],
    *values: str,
) -> tuple:
    r"""``(step, layer, *values)`` arrays of a long-format table, restricted to rows whose step and
    layer are finite (``layer`` as int). Missing value columns come back as NaN."""
    steps = _np(table["step"]).astype(float).reshape(-1)
    layers = _np(table["layer"]).astype(float).reshape(-1)
    keep = np.isfinite(steps) & np.isfinite(layers)
    out = [steps[keep], layers[keep].astype(int)]
    for name in values:
        column = (_np(table[name]).astype(float).reshape(-1) if name in table
                  else np.full(steps.size, np.nan))
        out.append(column[keep])
    return tuple(out)


def _residual_ratio_label(additive: bool) -> Tuple[str, str]:
    r"""``(name, symbol)`` of the ``residual_ratio.csv`` quantity.

    :math:`\lVert m_i\rVert/\lVert\mu_i\rVert` when the layer update is additive (``mu + m``); without an
    additive residual (``vfe3.train._residual_update_is_additive`` False: the predictive-content kernel
    readout with ``attention_residual=False``) the probe records the displacement ratio
    :math:`\lVert\mu^{\rm out}_i-\mu_i\rVert/\lVert\mu_i\rVert` (audit 2026-10-10 #34).
    """
    return (("Residual ratio", r"$\|m_i\|/\|\mu_i\|$") if additive
            else ("Displacement ratio", r"$\|\mu^{\rm out}_i-\mu_i\|/\|\mu_i\|$"))


def _residual_ratio_training_panel(
    table:    Mapping[str, object],

    *,
    additive: bool = True,               # vfe3.train._residual_update_is_additive of the run
) -> Optional[Dict[str, object]]:
    r"""Dashboard panel of the per-layer median residual ratio over training.

    ``table`` is ``diagnostics/residual_ratio.csv`` as columns (step, layer, q10, q50, q90, ...).
    The plotted quantity is the per-token median over the fixed validation windows of
    :math:`\lVert m_i\rVert/\lVert\mu_i\rVert`, the attention message relative to the layer's input
    mean, or the displacement ratio when the update is not ``additive`` (:func:`_residual_ratio_label`).
    Returns None when the table has no finite median.
    """
    steps, layers, q50 = _long_rows(table, "q50")
    if not np.isfinite(q50).any():
        return None
    grid = np.unique(steps)
    data: Dict[str, object] = {"step": grid}
    series = []
    colors = _layer_colors(int(layers.max()) + 1)
    for layer in np.unique(layers):
        column = np.full(grid.size, np.nan)
        rows = layers == layer
        for step_value, value in zip(steps[rows], q50[rows]):
            column[np.searchsorted(grid, step_value)] = value
        key = f"residual_ratio_q50_layer{int(layer)}"
        data[key] = column
        series.append((key, f"layer {int(layer)}", colors[int(layer)]))
    return {"title": "{} {} (token median)".format(*_residual_ratio_label(additive)), "ylabel": "ratio",
            "logy": True, "data": data, "series": series}


def _readout_panels(
    covariance_frozen: bool,
    residual_panel:    Optional[Dict[str, object]],

    *,
    by_layer:          bool = False,         # x axis is the layer (per-layer figure), not the step
) -> List[Dict[str, object]]:
    r"""Panels shared by :func:`plot_readout_diagnostics` and the conditional per-layer figure.

    The normalized residual-ratio panel replaces the unnormalized displacement
    :math:`\lVert \mathrm{result}-\mu\rVert_F` when ``diagnostics/residual_ratio.csv`` exists; the
    receiver-covariance panels are dropped when the covariance is frozen at encode (they would plot
    the constant token-table row). Over training the headline columns read the last layer, so each is
    drawn beside its ``_layer_max`` column; the routing-cap panels add the per-(layer, head) maxima of
    the capped fraction, the fully clamped rows and the self weight, and the layers attaining them
    (audit 2026-10-09 round 2, item 23). On the per-layer figure the headline columns are that layer's.
    Under ``diagnose_canonical_free_energy`` two further panels show the canonical :math:`F` terms at the
    captured beliefs and the model router's distance from the canonical router :math:`\beta^*`.
    """
    last = "" if by_layer else ", last layer"
    panels: List[Dict[str, object]] = [
        {"title": "Pre-update routing entropy", "ylabel": "nats", "series": [
            ("attn_entropy", "mean row entropy", _CB[0]),
            ("attn_entropy_min", "minimum scorable row", _CB[1])]},
    ]
    if residual_panel is not None:
        panels.append(residual_panel)
    else:
        panels.append({"title": r"Unnormalized displacement $\|\mathrm{out}-\mu\|_F$",
                       "ylabel": "L2 norm (no denominator)", "series": [
                           ("relational_mean_displacement_norm", "whole sequence", _CB[0])]})
    if not covariance_frozen:
        panels += [
            {"title": "Receiver covariance rank", "ylabel": "effective rank", "series": [
                ("effective_rank", "receiver bookkeeping", _CB[2])]},
            {"title": "Receiver covariance conditioning", "ylabel": "condition number", "series": [
                ("belief_cond_median", f"median{last}", _CB[3]),
                ("belief_cond_median_layer_max", "median, layer max", _CB[1])]},
        ]
    panels += [
        {"title": "Comparison covariance conditioning", "ylabel": "condition number", "logy": True,
         "series": [("relational_projected_cov_max_condition", f"maximum over Q and K{last}", _CB[1]),
                    ("relational_projected_cov_max_condition_layer_max", "maximum, layer max", _CB[3])]},
        {"title": "Routing cap", "ylabel": "fraction", "series": [
            ("relational_score_klmax_frac", f"capped entries{last}", _CB[0]),
            ("relational_score_klmax_frac_layer_max", "capped entries, layer max", _CB[5]),
            ("relational_score_klmax_frac_head_max", "capped entries, head max", _CB[4]),
            ("relational_clamped_row_frac_head_max", "fully clamped rows, head max", _CB[1]),
            ("relational_self_weight_head_max", r"mean self weight $\beta_{ii}$, head max", _CB[3]),
            ("relational_self_argmax_frac_head_max", "self-argmax rows, head max", _CB[2])]},
        {"title": "Routing cap location", "ylabel": "layer", "series": [
            ("relational_score_klmax_frac_argmax_layer", "capped entries", _CB[0]),
            ("relational_clamped_row_frac_argmax_layer", "fully clamped rows", _CB[1]),
            ("relational_self_lock_argmax_layer", "self weight", _CB[3])]},
        {"title": "Comparison map smallest singular value", "ylabel": "singular value", "series": [
            ("relational_score_q_raw_min_singular_value", "Q", _CB[0]),
            ("relational_score_k_raw_min_singular_value", "K", _CB[1])]},
        # diagnose_canonical_free_energy only (NaN otherwise, so the panels drop): the canonical F at the
        # beliefs each router read, which nothing on these routes optimizes (audit 2026-10-10 round 2, #24).
        {"title": "Canonical F at the captured beliefs (diagnostic)", "ylabel": "nats/token", "series": [
            ("total", "F at the layer input", _CB[7]),
            ("canonical_total_output", "F at the readout output", _CB[4]),
            ("self_coupling", r"self-coupling $\alpha D+R(\alpha)$", _CB[0]),
            ("self_divergence", r"self-divergence $D(q\|p)$", _CB[5]),
            ("belief_coupling", r"belief coupling $\sum_j\beta^*_{ij}D_{ij}$", _CB[1]),
            ("attention_entropy", "router entropy term", _CB[2])]},
        {"title": "Model router against the canonical router", "ylabel": "fraction of row mass", "series": [
            (key, label, _CB[(index + 3) % len(_CB)])
            for index, (key, label) in enumerate(_CANONICAL_ROUTER_SERIES)]},
    ]
    panels += [
        {"title": "Content-link spectrum (similarity invariant)", "ylabel": r"$|\lambda|$", "logy": True,
         "series": [("relational_link_spectral_radius_max", r"max head $\rho(T_hO_h)$", _CB[0]),
                    # vfe3.metrics.content_link_spectrum: at most SELFLOOP_RADIUS_SAMPLE (256) evenly spaced
                    # tokens (plus the D_i = I tokens), so the max bounds the all-token max from below.
                    ("relational_link_selfloop_spectral_radius_max",
                     "self-loop, max over a 256-token subsample (lower bound)", _CB[1]),
                    ("relational_link_selfloop_spectral_radius_mean", "self-loop, subsample mean", _CB[2]),
                    ("relational_link_min_abs_eigenvalue", r"min head $|\lambda|$", _CB[3])]},
        {"title": "Composition defect (spectral)", "ylabel": "distance",
         "series": [("relational_link_unit_distance_max", r"max $|\lambda-1|$", _CB[0]),
                    ("relational_link_idempotency_defect_max", r"max $|\lambda(\lambda-1)|$", _CB[1])]},
    ]
    return panels


@register_figure("readout_diagnostics")
def plot_readout_diagnostics(
    history: Dict,

    *,
    residual_ratio:    Optional[Mapping[str, object]] = None,   # diagnostics/residual_ratio.csv columns
    covariance_frozen: bool                           = False,
    residual_additive: bool                           = True,   # vfe3.train._residual_update_is_additive
    path:              Optional[str]                  = None,
) -> object:
    r"""Plot captured routing, the residual ratio, and Gaussian conditioning with no shared-F claim.

    With ``residual_ratio`` (the long-format table of :func:`read_long_table`) the displacement panel
    is the per-layer token-median ratio :math:`\lVert m_i\rVert/\lVert\mu_i\rVert` over training
    (titled as the displacement ratio when ``residual_additive`` is False, :func:`_residual_ratio_label`);
    without it the legacy unnormalized final-layer displacement is drawn and labeled as such.
    ``covariance_frozen`` drops the receiver-covariance panels (constant on that route).
    """
    residual_panel = (_residual_ratio_training_panel(residual_ratio, additive=residual_additive)
                      if residual_ratio is not None else None)
    return _history_dashboard(history, _readout_panels(covariance_frozen, residual_panel),
                              "Routing and readout diagnostics", path, ncols=2)


@register_figure("geometry_health")
def plot_geometry_health(
    history: Dict,                       # step + gauge/SPD/Fisher/guard health columns (any subset)

    *,
    family:         Optional[str] = None,
    path:           Optional[str] = None,
):
    r"""Gauge / SPD / Fisher geometry-health dashboard over training.

    Surfaces the converged-state geometry scalars ``diagnostics()`` already logs to ``metrics.csv`` but
    that no standard figure plotted: holonomy / cocycle flatness, gauge-frame magnitude and spread,
    belief-covariance conditioning and effective rank, belief Fisher precision, numerical-guard
    saturation, and attention-entropy collapse. Together they say whether the learned geometry
    stays meaningful rather than degenerating to a flat cocycle (phi -> 0, an UNGAUGED transformer) or
    a guard-pinned fixed point. Each panel self-gates on column presence."""
    fisher_label = pub_label("fisher_trace_mean", family=family)
    # The flat-transport closure panel (holonomy_wilson, cocycle_residual, holonomy_deviation) was removed:
    # no projected route writes those columns (audit 2026-10-12 round 2, item 13).
    panels = [
        {"title": "Gauge-frame spread", "ylabel": "spread", "series": [
            ("gauge_invariant_spread", "frame class-function spread", _CB[3]),
            ("gauge_head_logdet_spread", r"head $\log|\det|$", _CB[4])]},
        # block_glmc only: the U(1) phase of the head frames and the impropriety of the belief covariance against
        # the complex structure J (0 = proper); both columns are absent for every other group.
        {"title": "Complex structure (block_glmc)", "ylabel": "radians / [0, 1)", "series": [
            ("gauge_head_phase_abs_mean", r"mean $|\arg\det_{\mathbb{C}} U_h|$", _CB[0]),
            ("belief_circularity_mean", "belief circularity", _CB[3])]},
        {"title": "Gauge-frame magnitude", "ylabel": r"$\|\phi\|$", "series": [
            ("phi_norm_mean", r"mean $\|\phi\|$", _CB[0]),
            ("phi_norm_std", r"std $\|\phi\|$", _CB[5])]},
        {"title": "Embedded phi chart radius", "ylabel": r"$\|\phi^a G_a\|_F$", "logy": True,
         "series": [
            ("phi_matrix_norm_p95", "p95", _CB[0]),
            ("phi_matrix_norm_p99", "p99", _CB[1]),
            ("phi_matrix_norm_max", "max", _CB[3]),
            ("pos_phi_matrix_norm_p95", "positional p95", _CB[4]),
            ("pos_phi_matrix_norm_max", "positional max", _CB[5])]},
        {"title": "Projected M-step chart", "ylabel": "fraction / scale", "series": [
            ("phi_chart_projected_fraction", "projected fraction", _CB[1]),
            ("phi_chart_projection_scale_min", "minimum scale", _CB[2])]},
        {"title": "Projected M-step cost", "ylabel": "milliseconds", "series": [
            ("phi_chart_projection_ms", "projection time", _CB[0])]},
        {"title": "Vertex conditioning", "ylabel": r"$\kappa(\exp\phi)$", "logy": True,
         "series": [
            ("vertex_cond_median", "median", _CB[0]),
            ("vertex_cond_p95", "p95", _CB[1]),
            ("vertex_cond_p99", "p99", _CB[3])]},
        {"title": "Belief conditioning", "ylabel": r"$\kappa(\Sigma)$", "logy": True, "series": [
            ("belief_cond_p95", "p95", _CB[1]),
            ("belief_cond_max", "max", _CB[3])]},
        {"title": "Belief effective rank", "ylabel": r"$\mathrm{erank}(\Sigma)$", "series": [
            ("eff_rank_p5", "p5", _CB[2]),
            ("eff_rank_median", "median", _CB[0]),
            ("eff_rank_p95", "p95", _CB[4])]},
        {"title": "Belief precision", "ylabel": fisher_label,
         "logy": True, "series": [("fisher_trace_mean", fisher_label, _CB[3])]},
        {"title": "Guard saturation", "ylabel": "fraction", "series": [
            ("guard_sigma_floor_frac", r"$\sigma$ floor", _CB[0]),
            ("guard_sigma_ceil_frac", r"$\sigma$ ceil", _CB[1])]},
        {"title": "Numerical safety", "ylabel": "fraction", "series": [
            ("nonfinite_frac", "nonfinite", _CB[1])]},
        {"title": "Attention-entropy collapse", "ylabel": "count / nats", "series": [
            ("attn_entropy_collapsed_heads", "collapsed heads", _CB[0]),
            ("attn_entropy_min", "min row entropy", _CB[3])]},
    ]
    return _history_dashboard(history, panels, "Gauge / SPD / Fisher geometry health", path)


# The metrics.csv columns plot_geometry_health reads (the run finalizer passes exactly these).
GEOMETRY_HEALTH_COLUMNS = (
    "gauge_invariant_spread", "gauge_head_logdet_spread",
    "gauge_head_phase_abs_mean", "belief_circularity_mean",
    "phi_norm_mean", "phi_norm_std",
    "phi_matrix_norm_p95", "phi_matrix_norm_p99", "phi_matrix_norm_max",
    "pos_phi_matrix_norm_p95", "pos_phi_matrix_norm_max",
    "vertex_cond_median", "vertex_cond_p95", "vertex_cond_p99",
    "phi_chart_projected_fraction", "phi_chart_projection_scale_min",
    "phi_chart_projection_ms",
    "belief_cond_p95", "belief_cond_max", "eff_rank_p5", "eff_rank_median", "eff_rank_p95",
    "fisher_trace_mean", "guard_sigma_floor_frac", "guard_sigma_ceil_frac",
    "nonfinite_frac",
    "attn_entropy_min", "attn_entropy_collapsed_heads")


@register_figure("estep_quality")
def plot_estep_quality(
    history: Dict,                       # step + estep_f_drop / nondecreasing-frac / belief residuals

    *,
    path:    Optional[str] = None,
):
    r"""E-step inference-quality dashboard over training.

    Direct evidence for what the inner E-step contributes per eval: the free-energy drop across the
    inner loop (``estep_f_drop`` < 0 = F descended), the fraction of inner iterations that did NOT
    decrease F (a descent-quality readout for parallel mean-field -- EXPECTED nonzero, not a failure
    flag), and the last-iteration belief-change residuals for ``mu`` / ``Sigma`` / ``phi`` (the
    convergence certificate). Complements the standalone E-step convergence-trend curve."""
    panels = [
        {"title": "E-step free-energy drop", "ylabel": r"$F_{\mathrm{end}}-F_{\mathrm{start}}$",
         "series": [("estep_f_drop", "F drop", _CB[0])]},
        {"title": "Inner-loop monotonicity", "ylabel": "nondecreasing fraction",
         "series": [("estep_f_nondecreasing_frac", "nondecreasing frac", _CB[1])]},
        {"title": "Configured final movement", "ylabel": "step length", "logy": True, "series": [
            ("estep_r_mu_last", r"$q_{T-1}\to q_T$: $r_\mu$", _CB[0]),
            ("estep_r_sigma_last", r"$q_{T-1}\to q_T$: $r_\Sigma$", _CB[1]),
            ("estep_r_phi_last", r"$q_{T-1}\to q_T$: $r_\phi$", _CB[2])]},
        {"title": "One-step-ahead fixed-point residual", "ylabel": "residual", "logy": True,
         "series": [
            ("estep_fp_mu_rms", r"$q_T\to q_{T+1}$: $\mu$ RMS", _CB[0]),
            ("estep_fp_sigma_rms", r"$q_T\to q_{T+1}$: $\Sigma$ RMS", _CB[1]),
            ("estep_fp_phi_rms", r"$q_T\to q_{T+1}$: $\phi$ RMS", _CB[2]),
            ("estep_fp_kl", r"$\mathrm{KL}(q_{T+1}\Vert q_T)$", _CB[3])]},
    ]
    return _history_dashboard(history, panels, "E-step inference quality", path, ncols=4)


@register_figure("validation_sanity")
def plot_validation_sanity(
    history: Dict,                       # step + held-out generalization / positional / attention / geometry columns

    *,
    family: Optional[str] = None,
    path:    Optional[str] = None,
):
    r"""Held-out validation-sanity dashboard over training.

    The per-eval probes computed on a HELD-OUT batch (``_val_diagnostics``) that otherwise live only in
    ``metrics.csv``: the generalization gap, within-sequence positional loss and its tail/early ratio,
    causal-mask sanity (future leakage, row-sum error), positional-vs-content structure, structured-head
    masses and head redundancy, and the held-out gauge / SPD / Fisher geometry (the more credible
    counterpart to the train-batch ``geometry_health`` figure). Each panel self-gates on column
    presence and only live panels are laid out; eval-cadence columns draw only on their eval steps.

    The generalization gap is ``generalization_gap_fixed`` (validation CE minus the CE of a fixed
    training-split sample under the evaluation protocol) when the run logged it; otherwise the
    legacy single-training-batch ``generalization_gap``, titled as such because one batch makes it
    swing by tenths of a nat between evaluations."""
    fisher_label = pub_label("val_fisher_trace_mean", family=family)
    gap_ylabel = r"$\mathrm{CE}_{\mathrm{val}}-\mathrm{CE}_{\mathrm{train}}$"
    gap_panel = (
        {"title": "Generalization gap (fixed train sample)", "ylabel": gap_ylabel,
         "series": [("generalization_gap_fixed", "fixed sample", _CB[0])]}
        if _live(history, "generalization_gap_fixed") else
        {"title": "Generalization gap (one training batch)", "ylabel": gap_ylabel,
         "series": [("generalization_gap", "single batch", _CB[0])]})
    panels = [
        gap_panel,
        {"title": "Positional loss", "ylabel": "CE (nats)", "series": [
            ("pos_loss_first_q", "first quartile", _CB[0]),
            ("pos_loss_last_q", "last quartile", _CB[1])]},
        {"title": "Positional loss ratio", "ylabel": "last / first quartile",
         "series": [("pos_loss_ratio", "ratio", _CB[2])]},
        {"title": "Causal-mask sanity", "ylabel": "max error", "logy": True, "series": [
            ("val_future_leakage", "future leakage", _CB[0]),
            ("val_row_sum_error", "row-sum error", _CB[1])]},
        {"title": "Positional vs content", "ylabel": r"$R^2$",
         "series": [("val_pos_content_r2", "pos-content", _CB[3])]},
        {"title": "Structured heads", "ylabel": "mass / JS", "series": [
            ("val_prev_token_mass", "prev-token", _CB[0]),
            ("val_period_match_mass", "period-match", _CB[1]),
            ("val_head_redundancy_js", "redundancy JS", _CB[2])]},
        {"title": "Held-out gauge spread", "ylabel": "spread",
         "series": [("val_gauge_invariant_spread", "frame class-function spread", _CB[3])]},
        {"title": "Held-out conditioning", "ylabel": r"$\kappa(\Sigma)$", "logy": True,
         "series": [("val_belief_cond_p95", "p95", _CB[1])]},
        {"title": "Held-out precision", "ylabel": fisher_label,
         "logy": True, "series": [("val_fisher_trace_mean", fisher_label, _CB[3])]},
        {"title": "Held-out guard saturation", "ylabel": "fraction", "series": [
            ("val_guard_sigma_floor_frac", r"$\sigma$ floor", _CB[0]),
            ("val_guard_sigma_ceil_frac", r"$\sigma$ ceil", _CB[1])]},
        {"title": "Held-out frame norm", "ylabel": r"$\|\phi\|$", "series": [
            ("val_phi_norm_mean", r"mean $\|\phi\|$", _CB[0]),
            ("val_phi_norm_std", r"std $\|\phi\|$", _CB[5])]},
        # The held-out counterparts of geometry_health's chart-radius and vertex panels (audit 2026-10-16 F83).
        {"title": "Held-out phi chart radius", "ylabel": r"$\|\phi^a G_a\|_F$", "logy": True, "series": [
            ("val_phi_matrix_norm_p95", "p95", _CB[0]),
            ("val_phi_matrix_norm_p99", "p99", _CB[1]),
            ("val_phi_matrix_norm_max", "max", _CB[3])]},
        {"title": "Held-out vertex conditioning", "ylabel": r"$\kappa(\exp\phi)$", "logy": True,
         "series": [("val_vertex_cond_p99", "p99", _CB[3])]},
    ]
    return _history_dashboard(history, panels, "Held-out validation sanity", path, ncols=3)


VALIDATION_SANITY_COLUMNS = (
    "generalization_gap_fixed", "generalization_gap", "pos_loss_first_q", "pos_loss_last_q",
    "pos_loss_ratio", "val_future_leakage", "val_row_sum_error", "val_pos_content_r2",
    "val_prev_token_mass", "val_period_match_mass", "val_head_redundancy_js",
    "val_gauge_invariant_spread",
    "val_belief_cond_p95", "val_fisher_trace_mean", "val_guard_sigma_floor_frac",
    "val_guard_sigma_ceil_frac",
    "val_phi_norm_mean", "val_phi_norm_std",
    "val_phi_matrix_norm_p95", "val_phi_matrix_norm_p99", "val_phi_matrix_norm_max", "val_vertex_cond_p99")


@register_figure("optimizer_geometry")
def plot_optimizer_geometry(
    history: Dict,                       # step + per-group norms / update ratios, sigma_log moments, legacy role norms

    *,
    spikes:  Optional[Mapping[str, object]] = None,   # the same columns over divergence-spike rows
    path:    Optional[str] = None,
):
    r"""Optimizer geometry over training: parameter norms and update scales by parameter group.

    With the per-group columns, the parameter norms :math:`\lVert\theta_g\rVert_2` and the
    update-to-weight ratios :math:`\lVert\Delta\theta_g\rVert/\lVert\theta_g\rVert` are drawn for the
    per-type tables and the context-processing parameters. The log-variance table is excluded from
    both: under weight decay its norm decays toward 0, so a ratio with that denominator rises for a
    trivial reason. It is drawn instead as the mean and, in its own panel, the standard deviation of
    the stored :math:`\log\sigma^2` entries (``sigma_log_mean``, ``sigma_log_sd``). Lines are named at
    their right ends (``labels="direct"`` of :func:`_history_dashboard`), and each y range is fitted
    to its data. A run with only the legacy
    per-ROLE columns shows role norms and :math:`\lVert\nabla_\theta\rVert/\lVert\theta\rVert` for
    the 'mu' and 'phi' roles with what each role aggregates in a figure note, and the norm of the
    log-variance table labeled as a norm. The phi pullback-group panels (ridge-direction cosine,
    damped generalized condition, accepted factor scale, active rows, accepted chart norm) appear
    only when the run logged them, i.e. under the pullback-group phi update, and only then does the
    title name that update. The cosine is a reshaping diagnostic, not a basis-invariant angle.
    """
    hist = dict(history)
    panels: List[Dict[str, object]] = []
    grouped = any(_live(hist, f"param_norm_group_{group}") or _live(hist, f"update_ratio_group_{group}")
                  for group in PARAMETER_GROUPS)
    if grouped:
        norm_panels = _group_panels("param_norm_group_", r"$\|\theta_g\|_2$",
                                    exclude=("sigma_log_embed",), short=True)
        ratio_panels = _group_panels("update_ratio_group_", r"$\|\Delta\theta_g\|/\|\theta_g\|$",
                                     exclude=("sigma_log_embed",), short=True)
        for panel in norm_panels:
            panel["title"] = f"Parameter norm, {panel['title']}"
        for panel in ratio_panels:
            panel["title"] = f"Update-to-weight ratio, {panel['title']}"
        panels += norm_panels + ratio_panels
    else:
        for role in ("mu", "phi"):                               # synthesize the gradient-to-weight ratio
            gk, wk = f"grad_norm_{role}", f"weight_norm_{role}"
            if gk in hist and wk in hist:
                g = _np(hist[gk]).astype(float)
                w = _np(hist[wk]).astype(float)
                hist[f"grad_weight_ratio_{role}"] = np.where(w > 0, g / w, np.nan)
        panels += [
            {"title": "Role parameter norm", "ylabel": r"$\|\theta\|_2$", "logy": True, "series": [
                ("weight_norm_mu", "'mu' role", _CB[0]),
                ("weight_norm_phi", "'phi' role", _CB[2])]},
            {"title": "Gradient-to-weight ratio", "ylabel": r"$\|\nabla_\theta\|/\|\theta\|$",
             "logy": True, "series": [
                ("grad_weight_ratio_mu", "'mu' role", _CB[0]),
                ("grad_weight_ratio_phi", "'phi' role", _CB[2])]},
        ]
    if _live(hist, "sigma_log_mean") or _live(hist, "sigma_log_sd"):
        # One panel each: the standard deviation (about 0.05) is two orders of magnitude below the
        # mean (about 2), so on one linear axis it is a flat line at zero.
        panels += [
            {"title": "Stored log-variances: mean", "ylabel": r"mean of $\log\sigma^2$",
             "series": [("sigma_log_mean", "mean", _CB[1])]},
            {"title": "Stored log-variances: standard deviation", "ylabel": r"sd of $\log\sigma^2$",
             "series": [("sigma_log_sd", "standard deviation", _CB[3])]},
        ]
    else:
        panels.append({"title": r"Norm of the $\log\sigma^2$ table", "ylabel": r"$\|\log\sigma^2\|_2$",
                       "logy": True, "series": [
                           ("weight_norm_sigma", "decays toward 0 under weight decay", _CB[1])]})
    panels += [
        {"title": "Ridge-direction cosine",
         "ylabel": r"$\cos(v_\phi,g_\phi)$",
         "series": [("phi_ridge_direction_cosine_mean", "ridge direction", _CB[0])]},
        {"title": "Pullback conditioning", "ylabel": "damped generalized condition", "logy": True,
         "series": [
            ("phi_pullback_damped_gen_cond_median", "median", _CB[1]),
            ("phi_pullback_damped_gen_cond_max", "max", _CB[3])]},
        {"title": "Accepted factor scale", "ylabel": "scale", "series": [
            ("phi_group_trust_scale_mean", "mean", _CB[0]),
            ("phi_group_trust_scale_min", "min", _CB[1])]},
        {"title": "Factor-scale halvings", "ylabel": "halvings", "series": [
            ("phi_group_backtracks_mean", "mean", _CB[0]),
            ("phi_group_backtracks_max", "max", _CB[3])]},
        {"title": "Pullback active rows", "ylabel": "rows", "series": [
            ("phi_group_active_rows", "active", _CB[2])]},
        {"title": "Accepted chart norm", "ylabel": r"$\|\phi\|_F$", "series": [
            ("phi_group_chart_norm_max", "max", _CB[3])]},
    ]
    pullback = any(_live(hist, key) for key in _PULLBACK_COLUMNS)
    title = ("Optimizer geometry with pullback-group phi update" if pullback
             else "Optimizer geometry")
    return _history_dashboard(hist, panels, title, path, ncols=2,
                              note=None if grouped else _ROLE_NOTE, labels="direct", spikes=spikes)


# ---------------------------------------------------------------------------
# Figures of the eval-cadence diagnostics (metrics.csv M3 columns and the long-format
# diagnostics/residual_ratio.csv and diagnostics/attention_stats.csv tables).
# ---------------------------------------------------------------------------

def _count_label(value: float, _pos: object = None) -> str:
    r"""Compact count tick label: 4.9e8 -> '490M', 1.2e9 -> '1.2B', 2.4e4 -> '24k'."""
    value = float(value)
    for scale, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(value) >= scale:
            scaled = value / scale
            return f"{scaled:.0f}{suffix}" if abs(scaled) >= 10 else f"{scaled:.1f}{suffix}"
    return f"{value:.0f}"


def _interp_on_step(
    step:   np.ndarray,
    values: np.ndarray,
    query:  np.ndarray,
) -> np.ndarray:
    r"""Evaluate a monotone per-row column (``tokens_seen``, ``wall_clock_s``) at ``query`` steps.

    Linear interpolation over the finite ``(step, value)`` pairs, so an eval row that did not carry
    the column is placed between its log-cadence neighbors; one finite pair scales proportionally
    with the step. NaN when the column has no finite value.
    """
    keep = np.isfinite(step) & np.isfinite(values)
    if not keep.any():
        return np.full(query.shape, np.nan)
    xs, ys = step[keep], values[keep]
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    if xs.size == 1:
        return query * (ys[0] / xs[0]) if xs[0] != 0 else np.full(query.shape, ys[0])
    return np.interp(query, xs, ys, left=np.nan, right=np.nan)


@register_figure("learning_curves")
def plot_learning_curves(
    history: Mapping[str, object],       # metrics.csv columns (read_metrics_columns)

    *,
    tokens_per_step: Optional[float] = None,   # fallback token axis for runs without tokens_seen
    path:            Optional[str]   = None,
):
    r"""M3 learning curves: held-out and training-sample CE against training tokens and wall-clock.

    Panel A: validation CE (``val_ce``) and, when logged, the CE of the fixed training-split sample
    under the evaluation protocol (``train_sample_ce``) against cumulative training tokens
    (``tokens_seen``; for older runs ``step x tokens_per_step`` when given, else the step, labeled as
    such). Panel B: the same curves against wall-clock hours (``wall_clock_s``). Panel C: the fixed
    generalization gap :math:`\mathrm{CE}_{\mathrm{val}}-\mathrm{CE}_{\mathrm{train\,sample}}`
    (``generalization_gap_fixed``). Each panel appears only when its columns exist; completed corpus
    passes (``corpus_pass``) are marked by dotted lines. Raises :class:`FigureSkipped` when no
    finite ``val_ce`` exists.
    """
    step = _np(history["step"]).astype(float)

    def _column(name: str) -> np.ndarray:
        return (_np(history[name]).astype(float) if name in history
                else np.full(step.shape, np.nan))

    val = _column("val_ce")
    eval_rows = np.isfinite(val) & np.isfinite(step)
    if not eval_rows.any():
        raise FigureSkipped("metrics.csv has no finite val_ce")
    train = _column("train_sample_ce")
    gap = _column("generalization_gap_fixed")
    if _live(history, "tokens_seen"):
        tokens = _interp_on_step(step, _column("tokens_seen"), step)
        token_label = "training tokens"
    elif tokens_per_step:
        tokens = step * float(tokens_per_step)
        token_label = "training tokens (step x tokens per step)"
    else:
        tokens, token_label = step, "training step"
    hours = _interp_on_step(step, _column("wall_clock_s"), step) / 3600.0
    pass_steps = np.array([], dtype=float)
    corpus_pass = _column("corpus_pass")
    finite_pass = np.isfinite(corpus_pass) & np.isfinite(step)
    if finite_pass.any():
        passes, pass_step = np.floor(corpus_pass[finite_pass]), step[finite_pass]
        pass_steps = pass_step[1:][passes[1:] > passes[:-1]]

    panels = [("tokens", tokens, token_label)]
    if np.isfinite(hours[eval_rows]).any():
        panels.append(("hours", hours, "wall-clock hours"))
    has_gap = bool(np.isfinite(gap).any())
    with _compact_style():
        n_panels = len(panels) + (1 if has_gap else 0)
        fig, axes = plt.subplots(1, n_panels, squeeze=False,
                                 figsize=(MAX_FIGURE_WIDTH_IN if n_panels > 1 else 3.6, 2.6))
        axes = axes[0]

        def _passes(ax, x_of_step: np.ndarray) -> None:
            known = np.isfinite(x_of_step) & np.isfinite(step)
            if not known.any():
                return
            for boundary in pass_steps:
                x = np.interp(boundary, step[known], x_of_step[known])
                ax.axvline(x, color="#777777", ls=":", lw=0.7, alpha=0.6, zorder=0)

        for ax, (kind, x_axis, xlabel) in zip(axes, panels):
            keep = eval_rows & np.isfinite(x_axis)
            ax.plot(x_axis[keep], val[keep], "o-", ms=2.5, lw=1.2, color=_CB[0],
                    label=f"validation (last {val[keep][-1]:.3f})" if keep.any() else "validation")
            train_rows = np.isfinite(train) & np.isfinite(x_axis)
            if train_rows.any():
                ax.plot(x_axis[train_rows], train[train_rows], "s--", ms=2.5, lw=1.2,
                        color=_CB[1], label=f"fixed training sample (last {train[train_rows][-1]:.3f})")
            _passes(ax, x_axis)
            ax.set(xlabel=xlabel, ylabel="CE (nats/token)",
                   title="CE vs training tokens" if kind == "tokens" else "CE vs wall-clock time")
            if kind == "tokens" and token_label != "training step":
                ax.xaxis.set_major_formatter(FuncFormatter(_count_label))
            elif kind == "tokens":
                _step_xaxis(ax)
            ax.legend(frameon=False, loc="upper right")
        if has_gap:
            ax = axes[-1]
            keep = np.isfinite(gap) & np.isfinite(tokens)
            ax.plot(tokens[keep], gap[keep], "o-", ms=2.5, lw=1.2, color=_CB[2],
                    label=f"gap (last {gap[keep][-1]:.3f})" if keep.any() else "gap")
            ax.axhline(0.0, color="#444444", lw=0.7, ls="--")
            _passes(ax, tokens)
            ax.set(xlabel=token_label, ylabel=r"$\mathrm{CE}_{\mathrm{val}}-\mathrm{CE}_{\mathrm{train}}$",
                   title="Generalization gap (fixed sample)")
            if token_label != "training step":
                ax.xaxis.set_major_formatter(FuncFormatter(_count_label))
            else:
                _step_xaxis(ax)
            ax.legend(frameon=False, loc="best")
        fig.suptitle("Learning curves" + (" (dotted: completed corpus passes)" if pass_steps.size
                                          else ""))
        fig.tight_layout()
        return _save(fig, path)


_ATTENTION_STATS = (
    ("entropy_norm", r"normalized entropy $H(\beta_{i\cdot})/\log n_i$", (0.0, 1.0)),
    # ``{\sum}_j`` sets the index beside the sum; mathtext's ``\sum_j`` stacks it underneath, where
    # it runs into the top edge of the panel below the title.
    ("distance_mean", r"mean distance ${\sum}_j\beta_{ij}|i-j|$ (tokens)", None),
    ("self_weight", r"self-weight $\beta_{ii}$", (0.0, 1.0)),
    ("prior_overlap", r"prior overlap ${\sum}_j\min(\beta_{ij},\beta^{\mathrm{prior}}_{ij})$", (0.0, 1.0)),
)


def _layer_head_grid(
    table: Mapping[str, object],
    name:  str,
    rows:  np.ndarray,                   # boolean row mask (one evaluation step)
) -> np.ndarray:
    r"""The (L, H) matrix of statistic ``name`` over the masked rows (NaN where absent)."""
    layers = _np(table["layer"]).astype(float)[rows].astype(int)
    heads = _np(table["head"]).astype(float)[rows].astype(int)
    values = _np(table[name]).astype(float)[rows]
    grid = np.full((int(layers.max()) + 1, int(heads.max()) + 1), np.nan)
    grid[layers, heads] = values
    return grid


@register_figure("attention_stats")
def plot_attention_stats(
    table: Mapping[str, object],         # diagnostics/attention_stats.csv columns

    *,
    step:  Optional[float] = None,       # evaluation step to show (default: the last one)
    path:  Optional[str]   = None,

    specs:  Sequence[Tuple[str, str, Optional[Tuple[float, float]]]] = _ATTENTION_STATS,
    title:  str = "Attention statistics by layer and head",
    source: str = "attention_stats.csv",
    note:   Optional[str] = None,        # one-line figure footnote
):
    r"""Per-layer, per-head attention statistics at one evaluation (S2 panel B), as heatmaps.

    From ``diagnostics/attention_stats.csv`` (mean over causal query rows with at least two sources,
    on the fixed validation windows): the normalized entropy :math:`H(\beta_{i\cdot})/\log n_i`, the
    mean attention distance :math:`\sum_j\beta_{ij}|i-j|`, the self-weight :math:`\beta_{ii}`, and the
    overlap :math:`\sum_j\min(\beta_{ij},\beta^{\mathrm{prior}}_{ij})` with the attention of the
    log-prior alone (1 means the prior explains the pattern). Bounded statistics share the [0, 1]
    color range so layers and runs compare directly. ``specs`` (column, label, color bounds),
    ``title``, ``source`` and ``note`` let another long ``(step, layer, head)`` table reuse the layout
    (:func:`plot_routing_heads`). NaN cells are left blank and do not set the color range.
    """
    steps = _np(table["step"]).astype(float)
    finite_steps = steps[np.isfinite(steps)]
    if not finite_steps.size:
        raise FigureSkipped(f"{source} has no rows")
    shown = float(finite_steps.max()) if step is None else float(step)
    rows = (steps == shown) & np.isfinite(_np(table["layer"]).astype(float)) \
        & np.isfinite(_np(table["head"]).astype(float))
    stats = [spec for spec in specs
             if spec[0] in table and np.isfinite(_np(table[spec[0]]).astype(float)[rows]).any()]
    if not stats:
        raise FigureSkipped(f"{source} has no finite statistic at step {shown:.0f}")
    grids = [(spec, _layer_head_grid(table, spec[0], rows)) for spec in stats]
    n_layers, n_heads = grids[0][1].shape
    annotate = n_layers * n_heads <= 48
    n_cols = 2 if len(grids) > 1 else 1
    n_rows_fig = (len(grids) + n_cols - 1) // n_cols
    panel_height = max(1.7, 0.28 * n_layers + 0.9)
    with _compact_style():
        fig, axes = plt.subplots(n_rows_fig, n_cols, squeeze=False,
                                 figsize=(MAX_FIGURE_WIDTH_IN if n_cols > 1 else 3.6,
                                          panel_height * n_rows_fig + 0.4))
        for index, ((name, label, bounds), grid) in enumerate(grids):
            ax = axes[index // n_cols][index % n_cols]
            ax.grid(False)
            vmin, vmax = bounds if bounds is not None else (np.nanmin(grid), np.nanmax(grid))
            image = ax.imshow(grid, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
            fig.colorbar(image, ax=ax, shrink=0.85, pad=0.02)
            ax.set_title(label)
            ax.set(xticks=range(n_heads), yticks=range(n_layers), xlabel="head", ylabel="layer")
            if annotate:
                span = (vmax - vmin) or 1.0
                for layer in range(n_layers):
                    for head in range(n_heads):
                        value = grid[layer, head]
                        if np.isfinite(value):
                            ax.text(head, layer, f"{value:.2f}" if abs(value) < 100 else f"{value:.0f}",
                                    ha="center", va="center", fontsize=6,
                                    color="white" if (value - vmin) / span < 0.6 else "black")
        for index in range(len(grids), n_rows_fig * n_cols):
            axes[index // n_cols][index % n_cols].axis("off")
        n_query = _np(table["n_rows"]).astype(float)[rows] if "n_rows" in table else np.array([])
        suffix = (f", {int(np.nanmedian(n_query)):,} query rows per head"
                  if n_query.size and np.isfinite(n_query).any() else "")
        fig.suptitle(f"{title} (step {shown:,.0f}{suffix})")
        if note:
            fig.text(0.01, 0.005, note, fontsize=6, color="#444444", ha="left", va="bottom")
            fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
        else:
            fig.tight_layout()
        return _save(fig, path)


@register_figure("attention_stats_training")
def plot_attention_stats_training(
    table: Mapping[str, object],         # diagnostics/attention_stats.csv columns

    *,
    path:  Optional[str] = None,

    specs:  Sequence[Tuple[str, str, Optional[Tuple[float, float]]]] = _ATTENTION_STATS,
    title:  str = "Attention statistics over training",
    source: str = "attention_stats.csv",
    note:   Optional[str] = None,        # one-line figure footnote
):
    r"""The attention statistics of :func:`plot_attention_stats` over training.

    For each statistic and layer, the line is the mean over heads and the band spans the heads'
    minimum and maximum, so both the layer trend and the spread between heads are visible; NaN cells
    enter neither. Raises :class:`FigureSkipped` with fewer than two evaluation steps. ``specs``,
    ``title``, ``source`` and ``note`` as in :func:`plot_attention_stats`.
    """
    steps = _np(table["step"]).astype(float)
    grid_steps = np.unique(steps[np.isfinite(steps)])
    if grid_steps.size < 2:
        raise FigureSkipped(f"{source} has fewer than two evaluation steps")
    stats = [spec for spec in specs
             if spec[0] in table and np.isfinite(_np(table[spec[0]]).astype(float)).any()]
    if not stats:
        raise FigureSkipped(f"{source} has no finite statistic")
    layers = _np(table["layer"]).astype(float)
    layer_ids = np.unique(layers[np.isfinite(layers)]).astype(int)
    colors = _layer_colors(int(layer_ids.max()) + 1)
    n_cols = 2 if len(stats) > 1 else 1
    n_rows_fig = (len(stats) + n_cols - 1) // n_cols
    with _compact_style():
        fig, axes = plt.subplots(n_rows_fig, n_cols, squeeze=False,
                                 figsize=(MAX_FIGURE_WIDTH_IN if n_cols > 1 else 3.6,
                                          2.2 * n_rows_fig + 0.4))
        for index, (name, label, bounds) in enumerate(stats):
            ax = axes[index // n_cols][index % n_cols]
            values = _np(table[name]).astype(float)
            for layer in layer_ids:
                mean, low, high = [], [], []
                for step_value in grid_steps:
                    chunk = values[(steps == step_value) & (layers == layer)]
                    chunk = chunk[np.isfinite(chunk)]
                    mean.append(chunk.mean() if chunk.size else np.nan)
                    low.append(chunk.min() if chunk.size else np.nan)
                    high.append(chunk.max() if chunk.size else np.nan)
                ax.plot(grid_steps, mean, "o-", ms=2, lw=1.1, color=colors[layer],
                        label=f"layer {layer}")
                ax.fill_between(grid_steps, low, high, color=colors[layer], alpha=0.15, lw=0)
            if bounds is not None:
                ax.set_ylim(*bounds)
            ax.set(xlabel="training step", title=label)
            if _set_finite_xlim(ax, grid_steps):
                _step_xaxis(ax)
            if index == 0:
                ax.legend(frameon=False, loc="best", ncol=2 if layer_ids.size > 4 else 1)
        for index in range(len(stats), n_rows_fig * n_cols):
            axes[index // n_cols][index % n_cols].axis("off")
        fig.suptitle(f"{title} (line: head mean; band: head range)")
        if note:
            fig.text(0.01, 0.005, note, fontsize=6, color="#444444", ha="left", va="bottom")
            fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
        else:
            fig.tight_layout()
        return _save(fig, path)


# Per-(layer, head) routing-cap and holonomy statistics (routing_head_table): (column, label, color bounds).
_ROUTING_HEAD_STATS: Tuple[Tuple[str, str, Optional[Tuple[float, float]]], ...] = (
    ("cap_fraction", "capped entries / causal entries", (0.0, 1.0)),
    ("clamped_row_fraction", "fully clamped rows / query rows", (0.0, 1.0)),
    ("log_defect_rms", r"routing holonomy: RMS $\log\rho_{ijk}$", None),
    ("prior_log_defect_rms", r"prior holonomy: RMS $\log\rho_{ijk}$", None),
)
_ROUTING_HEAD_SOURCE = "the per-head routing-cap and holonomy tables"
# Below this mean two-step path weight per query row a head routes almost no mass along any triangle, and its
# path-weighted holonomy RMS is a ratio of vanishing sums (audit 2026-10-10 round 2, #22: 27.31 at 1.5e-35).
ROUTING_HOLONOMY_MIN_PATH_MASS = 1e-6


def routing_head_table(
    cap:      Optional[Mapping[str, object]],     # diagnostics/cap_activation.csv columns
    clamped:  Optional[Mapping[str, object]],     # diagnostics/cap_clamped_rows.csv columns
    holonomy: Optional[Mapping[str, object]],     # diagnostics/routing_holonomy.csv columns
) -> Optional[Dict[str, np.ndarray]]:
    r"""One long ``(step, layer, head)`` table of the per-head routing-cap and holonomy diagnostics.

    ``cap_fraction = capped / entries`` (``cap_activation.csv``), ``clamped_row_fraction =
    clamped_rows / rows`` (``cap_clamped_rows.csv``; a row whose every source cost is capped, the
    only case in which the clamp changes :math:`\beta`), NaN at a zero denominator, and the
    path-weighted RMS of :math:`\log\rho_{ijk}` for the router and its prior alone
    (``routing_holonomy.csv``) with the router's mean two-step path mass ``path_mass``. The router's
    holonomy is NaN where ``path_mass`` is below :data:`ROUTING_HOLONOMY_MIN_PATH_MASS`. A key absent
    from one table is NaN in its columns. None when every table is None or has no row (audit
    2026-10-10 #64).
    """
    sources = ((cap, (("cap_fraction", "capped", "entries"),)),
               (clamped, (("clamped_row_fraction", "clamped_rows", "rows"),)),
               (holonomy, (("log_defect_rms", "log_defect_rms", None),
                           ("prior_log_defect_rms", "prior_log_defect_rms", None),
                           ("path_mass", "path_mass", None))))
    merged: Dict[Tuple[float, int, int], Dict[str, float]] = {}
    for table, columns in sources:
        if not table or any(key not in table for key in ("step", "layer", "head")):
            continue
        keys = [_np(table[key]).astype(float).reshape(-1) for key in ("step", "layer", "head")]
        keep = np.isfinite(keys[0]) & np.isfinite(keys[1]) & np.isfinite(keys[2])
        for name, numerator, denominator in columns:
            if numerator not in table:
                continue
            value = _np(table[numerator]).astype(float).reshape(-1)
            if denominator is not None:
                count = _np(table[denominator]).astype(float).reshape(-1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    value = np.where(count > 0, value / count, np.nan)
            for s, l, h, v in zip(keys[0][keep], keys[1][keep], keys[2][keep], value[keep]):
                merged.setdefault((float(s), int(l), int(h)), {})[name] = float(v)
    if not merged:
        return None
    order = sorted(merged)
    out: Dict[str, np.ndarray] = {"step": np.array([k[0] for k in order]),
                                  "layer": np.array([k[1] for k in order], dtype=float),
                                  "head": np.array([k[2] for k in order], dtype=float)}
    for name, _, _ in _ROUTING_HEAD_STATS:
        out[name] = np.array([merged[k].get(name, np.nan) for k in order])
    out["path_mass"] = np.array([merged[k].get("path_mass", np.nan) for k in order])
    out["log_defect_rms"][out["path_mass"] < ROUTING_HOLONOMY_MIN_PATH_MASS] = np.nan
    return out


def _low_path_mass_note(
    table: Mapping[str, object],
    rows:  np.ndarray,                   # bool mask of the table rows the figure draws
    where: str,                          # how the figure treats the withheld cells
) -> Optional[str]:
    r"""Footnote counting the (step, layer, head) cells whose routing holonomy was withheld for low path mass."""
    if "path_mass" not in table:
        return None
    n_low = int(np.sum(rows & (_np(table["path_mass"]).astype(float) < ROUTING_HOLONOMY_MIN_PATH_MASS)))
    if not n_low:
        return None
    return (f"Routing holonomy withheld for {n_low} (step, layer, head) cell(s) whose mean two-step path mass is "
            f"below {ROUTING_HOLONOMY_MIN_PATH_MASS:g} per query row ({where}).")


@register_figure("routing_heads")
def plot_routing_heads(
    table: Mapping[str, object],         # routing_head_table output

    *,
    step:  Optional[float] = None,       # evaluation step to show (default: the last one)
    path:  Optional[str]   = None,
):
    r"""Routing cap and holonomy per layer and head at one evaluation, as heatmaps.

    The columns of :func:`routing_head_table` in the layout of :func:`plot_attention_stats`: the
    capped share of causal entries, the fully clamped share of query rows (both on [0, 1]), and the
    routing and prior holonomy RMS of :math:`\log\rho_{ijk}`, on the fixed validation windows. A head
    whose path mass is below :data:`ROUTING_HOLONOMY_MIN_PATH_MASS` has a blank routing-holonomy cell
    that does not set the color range, and a footnote counts those cells.
    """
    steps = _np(table["step"]).astype(float)
    finite_steps = steps[np.isfinite(steps)]
    shown = (float(finite_steps.max()) if step is None and finite_steps.size
             else float("nan") if step is None else float(step))
    note = _low_path_mass_note(table, steps == shown, "blank cells")
    return plot_attention_stats(table, step=step, path=path, specs=_ROUTING_HEAD_STATS,
                                title="Routing cap and holonomy by layer and head",
                                source=_ROUTING_HEAD_SOURCE, note=note)


@register_figure("routing_heads_training")
def plot_routing_heads_training(
    table: Mapping[str, object],         # routing_head_table output

    *,
    path:  Optional[str] = None,
):
    r"""The statistics of :func:`plot_routing_heads` over training (line: head mean; band: head range).

    Routing-holonomy cells withheld for low path mass enter neither the head mean nor the head range."""
    steps = _np(table["step"]).astype(float)
    note = _low_path_mass_note(table, np.isfinite(steps), "excluded from the head mean and range")
    return plot_attention_stats_training(table, path=path, specs=_ROUTING_HEAD_STATS,
                                         title="Routing cap and holonomy over training",
                                         source=_ROUTING_HEAD_SOURCE, note=note)


@register_figure("residual_ratio")
def plot_residual_ratio(
    table: Mapping[str, object],         # diagnostics/residual_ratio.csv columns

    *,
    additive: bool          = True,      # vfe3.train._residual_update_is_additive of the run
    path:     Optional[str] = None,
):
    r"""Residual ratio per layer (S1 panel A) and the normalization change when the residual is removed.

    Panel A: the token quantiles of :math:`r_i=\lVert m_i\rVert/\lVert\mu_i\rVert`, the attention
    message :math:`m_i` (the layer's attention output minus its input mean) relative to the input
    mean :math:`\mu_i`, on the fixed validation windows: the median per layer with a 10th to 90th
    percentile band over training, or with error bars at a single evaluation. A ratio far above 1
    means the residual term :math:`\mu_i` is small next to the message. Without an ``additive``
    update the same column is the displacement ratio :math:`\lVert\mu^{\rm out}_i-\mu_i\rVert/\lVert\mu_i\rVert`
    and is titled so (:func:`_residual_ratio_label`). Panel B: the median relative
    change :math:`\lVert\mathrm{LN}(\mu_i+m_i)-\mathrm{LN}(m_i)\rVert/\lVert\mathrm{LN}(\mu_i+m_i)\rVert`
    of the block normalization output when the residual is removed; omitted when every value is NaN
    (no block normalization).
    """
    steps, layers, q10, q50, q90, ln_change = _long_rows(
        table, "q10", "q50", "q90", "ln_rel_change_q50")
    if not np.isfinite(q50).any():
        raise FigureSkipped("residual_ratio.csv has no finite median")
    layer_ids = np.unique(layers)
    colors = _layer_colors(int(layer_ids.max()) + 1)
    grid_steps = np.unique(steps)
    has_ln = bool(np.isfinite(ln_change).any())
    name, symbol = _residual_ratio_label(additive)
    with _compact_style():
        n_panels = 2 if has_ln else 1
        fig, axes = plt.subplots(1, n_panels, squeeze=False,
                                 figsize=(MAX_FIGURE_WIDTH_IN if n_panels > 1 else 3.6, 2.7))
        axes = axes[0]
        if grid_steps.size >= 2:
            for layer in layer_ids:
                rows = layers == layer
                order = np.argsort(steps[rows])
                x = steps[rows][order]
                axes[0].plot(x, q50[rows][order], "o-", ms=2, lw=1.1, color=colors[layer],
                             label=f"layer {layer}")
                axes[0].fill_between(x, q10[rows][order], q90[rows][order], color=colors[layer],
                                     alpha=0.15, lw=0)
                if has_ln:
                    axes[1].plot(x, ln_change[rows][order], "o-", ms=2, lw=1.1,
                                 color=colors[layer], label=f"layer {layer}")
            for ax in axes:
                ax.set_xlabel("training step")
                if _set_finite_xlim(ax, grid_steps):
                    _step_xaxis(ax)
            axes[0].set_title(f"{symbol}: median, 10-90% band")
        else:
            order = np.argsort(layers)
            x, mid = layers[order], q50[order]
            lower = np.clip(mid - q10[order], 0, None)
            upper = np.clip(q90[order] - mid, 0, None)
            axes[0].errorbar(x, mid, yerr=np.vstack([lower, upper]), fmt="o", ms=3, capsize=2,
                             color=_CB[0], label="median (bars: 10-90%)")
            if has_ln:
                axes[1].plot(x, ln_change[order], "o", ms=3, color=_CB[1])
            for ax in axes:
                ax.set(xlabel="layer", xticks=layer_ids)
            axes[0].set_title(f"{symbol} at step {grid_steps[0]:,.0f}")
        axes[0].set_yscale("log")
        axes[0].set_ylabel("ratio (log)")
        axes[0].legend(frameon=False, loc="best", ncol=2 if layer_ids.size > 4 else 1)
        if has_ln:
            positive = ln_change[np.isfinite(ln_change)]
            if (positive > 0).all():
                axes[1].set_yscale("log")
            axes[1].set_ylabel("relative change")
            axes[1].set_title(r"normalization output change without $\mu_i$ (median)")
        fig.suptitle(f"{name} per layer on the fixed validation windows")
        fig.tight_layout()
        return _save(fig, path)


@register_figure("ln3_symmetry_breaking")
def plot_ln3_symmetry_breaking(
    frozen:  Dict,                       # {step, val_ce, omega (N,N,K,K), beta (H,N,N)}
    learned: Dict,

    *,
    period:  int = 3,
    path:    Optional[str] = None,
):
    r"""F3: the gauge does the work -- ln(3) symmetry-breaking on the period-3 stream.

    Panel A: val CE vs step for the frozen (gauge off) and learned (gauge on) arms, with the
    analytic averaging floor CE = ln 3 drawn. Panel B: the directed transport asymmetry of the
    learned arm. Panel C: per-head period-3 / prev-token structure scores, frozen vs learned.
    """
    from vfe3 import metrics
    ln3 = float(np.log(3.0))
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.7))
    axes[0].plot(_np(frozen["step"]), _np(frozen["val_ce"]), color=_CB[1], lw=2, label="frozen (gauge off)")
    axes[0].plot(_np(learned["step"]), _np(learned["val_ce"]), color=_CB[2], lw=2, label="learned (gauge on)")
    axes[0].axhline(ln3, color="#444444", ls="--", lw=1.2, label=r"$\ln 3$ averaging floor")
    axes[0].set(xlabel="training step", ylabel="val CE (nats)", title="Order learning vs the averaging floor")
    axes[0].legend(fontsize=8, frameon=False)
    asym = _np(metrics.transport_asymmetry(learned["omega"]))
    vmax = float(np.abs(asym).max()) or 1.0
    im = axes[1].imshow(asym, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    fig.colorbar(im, ax=axes[1], shrink=0.8, label=r"$\|\Omega_{ij}-\Omega_{ji}\|_F$")
    axes[1].set(xlabel="key j", ylabel="query i", title="Transport asymmetry (learned)")
    fr = metrics.structured_head_scores(frozen["beta"], period=period)
    le = metrics.structured_head_scores(learned["beta"], period=period)
    labels = ["prev-token", f"period-{period}"]
    xs = np.arange(len(labels))
    axes[2].bar(xs - 0.2, [float(_np(fr["prev_token"]).mean()), float(_np(fr["period_match"]).mean())],
                width=0.4, color=_CB[1], label="frozen")
    axes[2].bar(xs + 0.2, [float(_np(le["prev_token"]).mean()), float(_np(le["period_match"]).mean())],
                width=0.4, color=_CB[2], label="learned")
    axes[2].set(xticks=xs, xticklabels=labels, ylabel="mean attention mass", title="Structured-head scores")
    axes[2].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, path)


@register_figure("belief_trajectories")
def plot_belief_trajectories(
    trace:       Dict,                   # e_step_belief_trace output (mu (T+1,N,K), ...)
    layer_trace: Optional[Dict] = None,  # across_layer_belief_trace output (d_ai, effective_rank)

    *,
    path:        Optional[str] = None,
):
    r"""F4: belief trajectories across E-step iterations (mean-space path) and across layers (SPD).

    Panel A: a shared-PCA 2-D quiver of the belief means as inference iterates (arrows mu_t ->
    mu_{t+1}, colored by token position). Panel B: the across-layer cumulative affine-invariant SPD
    geodesic distance and mean effective rank. The across-training axis is out of scope (needs
    per-checkpoint replay).
    """
    mu = _np(trace["mu"])                                         # (T+1, N, K)
    t1, n, k = mu.shape
    flat = mu.reshape(-1, k) - mu.reshape(-1, k).mean(0)
    _, _, vt = np.linalg.svd(flat, full_matrices=False)
    proj = (flat @ vt[:2].T).reshape(t1, n, 2)
    ncol = 2 if layer_trace is not None else 1
    fig, axes = plt.subplots(1, ncol, figsize=(5.0 * ncol, 4.2), squeeze=False)
    ax = axes[0][0]
    cmap = plt.cm.viridis(np.linspace(0, 1, n))
    for j in range(n):
        ax.plot(proj[:, j, 0], proj[:, j, 1], "-", color=cmap[j], lw=0.8, alpha=0.7)
        ax.quiver(proj[:-1, j, 0], proj[:-1, j, 1],
                  np.diff(proj[:, j, 0]), np.diff(proj[:, j, 1]),
                  angles="xy", scale_units="xy", scale=1, color=cmap[j], width=0.004, alpha=0.7)
    ax.scatter(proj[0, :, 0], proj[0, :, 1], s=14, color="#333333", marker="o", label="init")
    ax.set(xlabel="PC 1", ylabel="PC 2", title="Belief-mean path over E-step iterations")
    if layer_trace is not None:
        axb = axes[0][1]
        layers = np.arange(_np(layer_trace["d_ai"]).size)
        axb.plot(layers, _np(layer_trace["d_ai"]), "o-", color=_CB[0], label=r"$d_{AI}(\Sigma^0,\Sigma^l)$")
        axb.set(xlabel="layer", ylabel="cumulative SPD distance", title="Belief geometry vs depth")
        axc = axb.twinx()
        axc.plot(layers, _np(layer_trace["effective_rank"]), "s--", color=_CB[1])
        axc.set_ylabel("mean effective rank", color=_CB[1])
        axb.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    return _save(fig, path)


# --- linguistic-category labeling of gpt2-BPE tokens (category bucketing of plot_vocab_confusion) --

# A compact English function-word (stopword) set for the function/content taxonomy. Matched on the
# stripped, lower-cased decoded token, so a BPE word-start (" the") and a bare token ("the") both hit.
_STOPWORDS = frozenset((
    "a an the this that these those some any all each every no none "
    "i you he she it we they me him her us them my your his its our their mine yours hers ours theirs "
    "is are was were be been being am do does did doing have has had having "
    "will would shall should can could may might must ought "
    "of in on at by for with about against between into through during before after "
    "above below to from up down out off over under again further then once here there "
    "and or but nor so yet because as if while although though unless until than "
    "not only own same too very just more most other such "
    "what which who whom whose where when why how "
    "s t re ve ll d m"                                       # common BPE contraction fragments
).split())

_BPE_CAT_NAMES  = ["punctuation", "number", "word (lc)", "word (Cap)", "subword", "space/other"]
_FUNC_CAT_NAMES = ["punctuation", "number", "function", "content", "other"]


def _bpe_category(text: str) -> int:
    r"""Index into ``_BPE_CAT_NAMES`` from a decoded gpt2 token's STRUCTURE.

    Leverages BPE structure: a leading space marks a word-start, its absence a continuation subword
    (e.g. ``ing``). Categories: punctuation/symbol (no letters), number (all-digit), word-start
    lower-case, word-start Capitalized, continuation subword, whitespace/other.
    """
    if not text or text.isspace():
        return 5
    core = text.strip()
    if not core:
        return 5
    if core.isdigit():
        return 1
    if not any(c.isalpha() for c in core):
        return 0
    if text[:1] == " ":                                      # leading space -> a new word starts here
        return 3 if core[:1].isupper() else 2
    return 4                                                 # no leading space + letters -> subword piece


def _funccontent_category(text: str) -> int:
    r"""Index into ``_FUNC_CAT_NAMES``: punctuation, number, function-word, content-word, or other.

    Linguistic split on the stripped lower-cased token: a purely-alphabetic token is a FUNCTION word
    if it is in ``_STOPWORDS`` else a CONTENT word; non-alphabetic tokens fall to punctuation / number
    / other.
    """
    core = text.strip().lower()
    if not core:
        return 4
    if core.isdigit():
        return 1
    if not any(c.isalpha() for c in core):
        return 0
    if not core.isalpha():
        return 4
    return 2 if core in _STOPWORDS else 3


def _token_category_labels(
    token_ids: object,                   # (M,) token ids
    decode:    object,                   # decode(list[int]) -> str (e.g. the gpt2 tiktoken decoder)
    taxonomy:  str,                      # 'bpe' or 'function_content'
) -> tuple:                              # (labels (M,) int, names list[str])
    r"""Per-token category index under ``taxonomy`` (each unique id decoded + classified once)."""
    if taxonomy == "bpe":
        fn, names = _bpe_category, _BPE_CAT_NAMES
    elif taxonomy == "function_content":
        fn, names = _funccontent_category, _FUNC_CAT_NAMES
    else:
        raise ValueError(f"unknown taxonomy {taxonomy!r} (expected 'bpe' / 'function_content')")
    ids = _np(token_ids).astype(int)
    cat = {int(t): fn(decode([int(t)])) for t in np.unique(ids)}
    return np.array([cat[int(t)] for t in ids]), names


_EQUIVARIANCE_LABELS = {
    "routing_score":              r"score $D_{ij}^h$",
    "routing_weights":            r"weights $\beta_{ij}^h$",
    "readout_mean":               r"mean $U_i^{-1}m_i$",
    "readout_covariance":         r"cov. $U_i^{-1}\Sigma_i^+U_i^{-\top}$",
    "receiver_force":             r"force $U_i^\top\nabla_\mu F$",
    "receiver_update_mean":       r"update $U_i^{-1}\mu_i^+$",
    "receiver_update_covariance": r"update $U_i^{-1}\Sigma_i^+U_i^{-\top}$",
    "receiver_energy":            r"energy $F$",
    "self_coupling":              r"self term $\alpha_iD_i+R(\alpha_i)$",
}


@register_figure("gauge_equivariance")
def plot_gauge_equivariance(
    resid: Dict,                         # vfe3.viz.extract.projected_gauge_equivariance output

    *,
    path:  Optional[str] = None,
):
    r"""Gauge equivariance of the executed projected routing and readout.

    One bar per layer and quantity: the maximum relative deviation
    :math:`\max|x'-x|/\max|x|` under random passive frame actions
    :math:`(U_i,\mu_i,\Sigma_i)\mapsto(g_iU_i,\,g_i\mu_i,\,g_i\Sigma_ig_i^\top)`, computed in float64,
    on a log axis with the float64 unit roundoff as a reference line.
    """
    residuals = resid.get("residuals", {})
    keys = [key for key in _EQUIVARIANCE_LABELS if key in residuals]
    if not keys:
        raise FigureSkipped("projected_gauge_equivariance returned no residuals")
    n_layers = max(len(residuals[key]) for key in keys)
    width = 0.8 / max(n_layers, 1)
    floor = float(resid.get("unit_roundoff", 2.220446049250313e-16))
    with _compact_style():
        fig, ax = plt.subplots(figsize=(MAX_FIGURE_WIDTH_IN, 2.8))
        x = np.arange(len(keys))
        for layer in range(n_layers):
            values = np.array([residuals[key][layer] if layer < len(residuals[key]) else np.nan
                               for key in keys], dtype=float)
            # A bitwise-exact quantity has zero deviation; draw it at the reference floor.
            shown = np.where(values > 0, values, floor)
            ax.bar(x + (layer - (n_layers - 1) / 2) * width, shown, width,
                   color=_CB[layer % len(_CB)], label=f"layer {layer}")
        ax.axhline(floor, color="#444444", ls=":", lw=1, label="float64 unit roundoff")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([_EQUIVARIANCE_LABELS[key] for key in keys], rotation=20, ha="right")
        ax.set_ylabel("max relative deviation")
        ax.set_title(f"Projected gauge equivariance ({resid.get('readout_mode', '')}, "
                     f"{resid.get('n_draws', '?')} frame draws, float64)")
        ax.legend(fontsize=6, frameon=False, ncol=min(n_layers + 1, 4))
        fig.tight_layout()
    return _save(fig, path)


_MAP_ORDER = ("S_Q", "S_K", "T", "O")
_MAP_LABELS = {"S_Q": r"$S_{Q,h}$", "S_K": r"$S_{K,h}$", "T": r"$T_h$", "O": r"$O_h$"}


@register_figure("head_block_norms")
def plot_head_block_norms(
    norms: Mapping[int, Mapping[str, object]],   # layer -> {"S_Q","S_K","T","O"} -> (H, B) Frobenius norms

    *,
    path:  Optional[str] = None,
):
    r"""Head x frame-block energy of the comparison and content maps, per layer (S2 panel A).

    ``norms`` is :func:`vfe3.posthoc.maps.head_block_norms` output: for each layer and each map
    :math:`M\in\{S_{Q,h},S_{K,h},T_h,O_h\}`, the :math:`H\times B` matrix of Frobenius norms
    :math:`n_{hb}=\lVert M_h|_b\rVert_F` of head :math:`h`'s map restricted to frame block
    :math:`b`'s coordinates. Each cell shows the block's share of the head's energy,

    .. math:: s_{hb}=\frac{n_{hb}^2}{\sum_{b'}n_{hb'}^2},

    so every row sums to one and a head that reads all blocks evenly has :math:`s_{hb}=1/B`. When
    :math:`H=B` the panel title gives the range of the diagonal (own-block) shares, which is the
    pairing the retired ``gauge_head_specialization`` figure assumed without measuring it.
    """
    layers = sorted(int(layer) for layer in norms)
    if not layers:
        raise FigureSkipped("head_block_norms returned no layers")
    maps = [name for name in _MAP_ORDER if any(name in norms[layer] for layer in layers)]
    if not maps:
        raise FigureSkipped("head_block_norms returned none of S_Q, S_K, T, O")
    shares: Dict[tuple, np.ndarray] = {}
    for layer in layers:
        for name in maps:
            if name not in norms[layer]:
                continue
            energy = np.square(_np(norms[layer][name]).astype(float))       # (H, B)
            total = energy.sum(axis=1, keepdims=True)
            shares[(layer, name)] = np.divide(energy, total, out=np.full_like(energy, np.nan),
                                              where=total > 0)
    n_heads, n_blocks = next(iter(shares.values())).shape
    finite_max = [float(np.nanmax(value)) for value in shares.values() if np.isfinite(value).any()]
    vmax = max(finite_max) if finite_max else 1.0
    annotate = n_heads * n_blocks <= 36
    with _compact_style():
        fig, axes = plt.subplots(len(layers), len(maps), squeeze=False,
                                 figsize=(MAX_FIGURE_WIDTH_IN, 1.35 * len(layers) + 1.0))
        image = None
        for row, layer in enumerate(layers):
            for col, name in enumerate(maps):
                ax = axes[row][col]
                ax.grid(False)
                if (layer, name) not in shares:
                    ax.axis("off")
                    continue
                share = shares[(layer, name)]
                image = ax.imshow(share, cmap="viridis", vmin=0.0, vmax=vmax, aspect="auto")
                title = f"layer {layer}, {_MAP_LABELS[name]}"
                if share.shape[0] == share.shape[1]:
                    diagonal = np.diag(share)
                    title += f" (own {np.nanmin(diagonal):.2f}-{np.nanmax(diagonal):.2f})"
                ax.set_title(title, fontsize=6.5)
                ax.set_xticks(range(share.shape[1]))
                ax.set_yticks(range(share.shape[0]))
                ax.tick_params(labelsize=6, length=1.5)
                if row == len(layers) - 1:
                    ax.set_xlabel("frame block b")
                if col == 0:
                    ax.set_ylabel("head h")
                if annotate:
                    for h in range(share.shape[0]):
                        for b in range(share.shape[1]):
                            if np.isfinite(share[h, b]):
                                ax.text(b, h, f"{share[h, b]:.2f}", ha="center", va="center",
                                        fontsize=6, color="white" if share[h, b] < 0.6 * vmax
                                        else "black")
        fig.suptitle(f"Head \u00d7 frame-block energy share of the interface maps "
                     f"(row sums 1; uniform 1/B = {1.0 / n_blocks:.2f})")
        fig.tight_layout()
        if image is not None:
            fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02,
                         label=r"$\|M_h|_b\|_F^2/\|M_h\|_F^2$")
        return _save(fig, path)


@register_figure("attention_structure")
def plot_attention_structure(
    beta: object,                        # (L,H,N,N) or (H,N,N) attention weights

    *,
    path: Optional[str] = None,
):
    r"""F8: attention structure -- per-(layer,head) entropy, redundancy, and distance decay.

    For a multi-layer model the input is (L, H, N, N); the leading (layer, head) axes are flattened
    to L*H curves but each keeps its identity as ``L{l}H{h}`` (NOT renumbered 0..L*H-1), so depth is
    legible. Panel A: per-(layer,head) row-entropy violins vs the uniform log N reference (does
    attention sharpen). Panel B: the (layer,head)x(layer,head) Jensen-Shannon redundancy heatmap
    (which maps are specialized vs redundant, across layers too). Panel C: the per-(layer,head)
    attention-vs-offset profile on a log axis (positional decay). A single-layer (H, N, N) input
    falls back to plain ``h{h}`` labels.
    """
    from vfe3 import metrics
    import torch
    b = beta if hasattr(beta, "dim") else torch.as_tensor(_np(beta))
    if b.dim() == 4:                                             # (L, H, N, N) -> (L*H, N, N), KEEP (layer,head)
        L, H = int(b.shape[0]), int(b.shape[1])                  # identity: labels track the reshape C-order
        labels = [f"L{l}H{hh}" for l in range(L) for hh in range(H)]   # L0H0, L0H1, L1H0, ... matches reshape(-1)
        b = b.reshape(-1, b.shape[-2], b.shape[-1])
    else:                                                        # (H, N, N) single layer -> plain head labels
        labels = [f"h{hh}" for hh in range(int(b.shape[0]))]
    rows = _np(metrics.attention_entropy_rows(b))               # (L*H, N)
    n = b.shape[-1]
    h = b.shape[0]
    _rot = 90 if h > 8 else 0                                    # rotate only crowded (layer,head) ticks
    _tick_size = 6 if h > 8 else 8
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))
    axes[0].violinplot([rows[i] for i in range(h)], showmeans=True)
    axes[0].axhline(float(np.log(n)), color="#444444", ls="--", lw=1, label=r"uniform $\log N$")
    axes[0].set(xticks=range(1, h + 1), ylabel="row entropy (nats)", title="Per-(layer,head) attention entropy")
    axes[0].set_xticklabels(labels, rotation=_rot, fontsize=_tick_size)
    axes[0].legend(fontsize=8, frameon=False)
    js = _np(metrics.head_redundancy_js(b))                     # (L*H, L*H) over all (layer,head) maps
    im = axes[1].imshow(js, cmap="magma", aspect="auto")
    fig.colorbar(im, ax=axes[1], shrink=0.8, label="JS divergence (nats)")
    axes[1].set(xticks=range(h), yticks=range(h), xlabel="layer-head", ylabel="layer-head", title="Head redundancy")
    axes[1].set_xticklabels(labels, rotation=_rot, fontsize=_tick_size)
    axes[1].set_yticklabels(labels, fontsize=_tick_size)
    dd = metrics.attention_distance_decay(b)
    prof = _np(dd["profile"])                                    # (L*H, N)
    off = _np(dd["offsets"])
    for i in range(h):
        # Below 1e-8 the profile is numerical zero: end the curve there instead of drawing the
        # floor as a plateau that stretches the log axis over decades of nothing.
        axes[2].plot(off, np.where(prof[i] > 1e-8, prof[i], np.nan), color=plt.cm.viridis(i / max(h - 1, 1)), lw=1.2, label=labels[i])
    axes[2].set_yscale("log")
    axes[2].set(xlabel=r"offset $|i-j|$", ylabel=r"mean $\beta$", title="Attention distance decay")
    axes[2].legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    return _save(fig, path)


@register_figure("per_layer_diagnostics")
def plot_per_layer_diagnostics(
    per_layer: Dict,                     # diagnostics_per_layer output: each value an (L,) sequence

    *,
    residual_ratio:    Optional[Mapping[str, object]] = None,   # diagnostics/residual_ratio.csv columns
    covariance_frozen: bool                           = False,
    residual_additive: bool                           = True,   # vfe3.train._residual_update_is_additive
    path:              Optional[str]                  = None,
):
    r"""F8b: per-LAYER (inference-depth) diagnostics -- the depth axis the stack collapses.

    The aggregate metrics.csv and every converged-state figure show only the FINAL block; this
    plots ``VFEModel.diagnostics_per_layer`` block-by-block so depth is legible: which layer carries
    the coupling vs entropy mass, and whether belief rank builds up with depth. Panel A: belief-channel free-energy budget (self-coupling, belief-coupling, attention
    entropy, total) vs layer. Panel B: gauge action (group-invariant spread, mean ``||phi||``) vs layer.
    Panel C: belief geometry (effective rank, attention entropy; condition number on a twin log axis) vs
    layer.

    On the conditional (readout) route the panels are those of :func:`plot_readout_diagnostics`
    with layer on the x axis. With ``residual_ratio`` the unnormalized displacement panel is replaced
    by the 10th, 50th and 90th token percentiles of :math:`\lVert m_i\rVert/\lVert\mu_i\rVert` per
    layer at the table's last evaluation step (titled as the displacement ratio when
    ``residual_additive`` is False); ``covariance_frozen`` drops the constant receiver-covariance panels.
    """
    if any(k.endswith("_semantics") for k in per_layer):
        n_layers = len(per_layer["attn_entropy"])
        residual_panel = None
        if residual_ratio is not None:
            steps, layers, q10, q50, q90 = _long_rows(residual_ratio, "q10", "q50", "q90")
            if steps.size and np.isfinite(q50).any():
                last = steps == steps.max()
                order = np.argsort(layers[last])
                data: Dict[str, object] = {"step": layers[last][order].astype(float)}
                for name, values in (("q10", q10), ("q50", q50), ("q90", q90)):
                    data[f"residual_ratio_{name}"] = values[last][order]
                residual_panel = {
                    "title": "{} {} (step {:,})".format(*_residual_ratio_label(residual_additive),
                                                         int(steps.max())),
                    "ylabel": "ratio", "logy": True, "data": data, "series": [
                        ("residual_ratio_q90", "90th percentile", _CB[3]),
                        ("residual_ratio_q50", "median", _CB[0]),
                        ("residual_ratio_q10", "10th percentile", _CB[5])]}
        fig = _history_dashboard({**per_layer, "step": np.arange(n_layers)},
                                 _readout_panels(covariance_frozen, residual_panel, by_layer=True),
                                 "Captured routing and readout by layer", None, ncols=2)
        with _compact_style():
            for ax in fig.axes:
                if ax.axison:
                    ax.set(xlabel="layer", xticks=np.arange(n_layers))
                    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:.0f}"))
            fig.tight_layout()
            return _save(fig, path)
    g = lambda k: _np(per_layer[k]).reshape(-1)
    L = g("self_coupling").size if "self_coupling" in per_layer else g(next(iter(per_layer))).size
    x = np.arange(L)
    _m = "o-" if L > 1 else "o"                                  # a single layer has no line to draw
    with _compact_style():
        fig, axes = plt.subplots(1, 3, figsize=(MAX_FIGURE_WIDTH_IN, 2.9))
        for k, c, lab in (("self_coupling", _CB[0], r"self $\alpha\,$KL$(q\|p)$"),
                          ("belief_coupling", _CB[1], r"belief $\sum\beta\,$KL"),
                          ("attention_entropy", _CB[2], "attn entropy term"),
                          ("total", _CB[7], "total (belief F)")):
            if k in per_layer:
                axes[0].plot(x, g(k), _m, color=c, lw=1.4, label=lab)
        axes[0].set(xticks=x, xlabel="layer", ylabel="nats", title="Free-energy budget by layer")
        axes[0].legend(fontsize=6, frameon=False)
        for k, c, lab in (("gauge_invariant_spread", _CB[1], "class-function spread"),
                          ("phi_norm_mean", _CB[2], r"mean $\|\phi\|$")):
            if k in per_layer:
                axes[1].plot(x, g(k), _m, color=c, lw=1.4, label=lab)
        axes[1].set(xticks=x, xlabel="layer", ylabel="magnitude", title="Gauge action by layer")
        axes[1].legend(fontsize=6, frameon=False)
        ax = axes[2]
        if "effective_rank" in per_layer:
            ax.plot(x, g("effective_rank"), _m, color=_CB[0], lw=1.4, label="effective rank")
        if "attn_entropy" in per_layer:
            ax.plot(x, g("attn_entropy"), ("s--" if L > 1 else "s"), color=_CB[2], lw=1.2, label="attn entropy (nats)")
        ax.set(xticks=x, xlabel="layer", ylabel="rank / nats", title="Belief geometry by layer")
        handles, labels = ax.get_legend_handles_labels()
        if "belief_cond_median" in per_layer:
            axr = ax.twinx()
            # Name the axis in the label: this marker lives on the right (log) condition-number axis, not
            # the left rank/nats axis -- without it the lone triangle reads as a second entropy series.
            axr.plot(x, g("belief_cond_median"), ("^:" if L > 1 else "^"), color=_CB[1], lw=1.2,
                     label="cond median (right, log)")
            axr.set_ylabel("condition number")
            axr.set_yscale("log")
            axr.grid(False)
            h2, l2 = axr.get_legend_handles_labels()
            handles += h2
            labels += l2
        ax.legend(handles, labels, fontsize=6, frameon=False, loc="upper left")
        fig.tight_layout()
        return _save(fig, path)


@register_figure("belief_spectrum")
def plot_belief_spectrum(
    sigma:     object,                   # (N, K) diagonal OR (N, K, K) full converged covariances

    *,
    eps:       float = 1e-6,
    sigma_max: Optional[float] = 5.0,
    diagonal:  Optional[bool]  = None,
    family:    Optional[str]   = None,
    path:      Optional[str]   = None,
):
    r"""F9: belief covariance geometry -- effective rank, guarded scree, and conditioning.

    Panel A: the per-token effective-rank violin (the distribution the logged mean discards).
    Panel B: the eigenvalue scree (median + 10-90 band, log axis) with the eps floor and, when
    enabled, sigma_max ceiling, exposing whether apparent rank collapse rides a guard. Panel C:
    the per-token spectral condition-number histogram.
    """
    from vfe3 import metrics
    sp = metrics.belief_spectrum(
        sigma,
        diagonal=diagonal,
        eps=eps,
        family=family,
    )
    eig = _np(sp["eigenvalues"])                                 # (N, K) descending
    rank = _np(sp["effective_rank"])
    cond = _np(sp["condition"])
    covariance_eps = eps
    covariance_max = sigma_max
    if family is not None:
        import torch
        from vfe3.families.base import get_family
        family_cls = get_family(family)
        covariance_eps = family_cls.covariance_floor(eps)
        if family_cls.cov_kind == "diagonal":
            if sigma_max is not None:
                covariance_max = float(family_cls.covariance_diagonal(
                    torch.tensor([sigma_max]),
                    eps=0.0,
                )[0])
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))
    axes[0].violinplot([rank], showmeans=True)
    axes[0].set(xticks=[1], xticklabels=["beliefs"], ylabel="effective rank",
                title="Per-token effective rank")
    kk = np.arange(1, eig.shape[1] + 1)
    _median_band(axes[1], kk, eig.T, _CB[0], "eigenvalue")
    positive = eig[np.isfinite(eig) & (eig > 0)]
    if positive.size and covariance_eps < float(positive.min()) * 1e-3:
        # Three or more decades below every eigenvalue: drawing it would squeeze the spectrum into
        # the top of the panel, so the legend states it instead.
        axes[1].plot([], [], color=_CB[1], ls="--", lw=1,
                     label=rf"$\varepsilon$ floor = {_plain_tick(covariance_eps)} (below the axis)")
    else:
        axes[1].axhline(covariance_eps, color=_CB[1], ls="--", lw=1,
                        label=r"$\varepsilon$ floor")
    if covariance_max is not None:
        axes[1].axhline(covariance_max, color=_CB[2], ls="--", lw=1,
                        label=r"$\sigma_{\max}$ ceiling")
    axes[1].set_yscale("log")
    axes[1].set(xlabel="eigenvalue index", ylabel="eigenvalue", title="Guarded spectrum (scree)")
    axes[1].legend(fontsize=8, frameon=False)
    axes[2].hist(cond, bins=24, color=_CB[0], alpha=0.85)
    axes[2].set(xlabel=r"condition number $\lambda_{\max}/\lambda_{\min}$", ylabel="tokens",
                title="Belief conditioning")
    fig.tight_layout()
    return _save(fig, path)


_ELLIPSE_MAX = 24                        # ellipses outlined by plot_spd_ellipses


@register_figure("spd_ellipses")
def plot_spd_ellipses(
    mu:       object,                    # (N, K) belief means
    sigma:    object,                    # (N, K) diagonal OR (N, K, K) full covariances

    *,
    dims:     tuple = (0, 1),
    diagonal: Optional[bool] = None,
    eps:      float          = 1e-12,
    family:   Optional[str]  = None,
    path:     Optional[str]  = None,
):
    r"""F9 (companion): correlation-bearing SPD covariance ellipses, colored by effective rank.

    For a full covariance the ellipse orientation and axes come from the eigendecomposition of the
    2x2 coordinate sub-block (the old ellipse used diagonal variances only, showing no correlation);
    a diagonal belief degrades to axis-aligned. The outline color encodes per-token effective rank;
    with more than :data:`_ELLIPSE_MAX` tokens a subset spread evenly over the rank order is
    outlined and every other token is shown by its mean only.
    """
    from matplotlib.patches import Ellipse
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from vfe3 import metrics
    m = _np(mu)
    covariance = sigma
    if family is not None:
        import torch
        from vfe3.families.base import get_family
        family_cls = get_family(family)
        dispersion = sigma if hasattr(sigma, "detach") else torch.as_tensor(sigma)
        if family_cls.cov_kind == "diagonal":
            covariance = family_cls.covariance_diagonal(dispersion)
    s = _np(covariance)
    a, b = dims
    is_full = s.ndim >= 3 and s.shape[-1] == s.shape[-2] if diagonal is None else (not diagonal)
    rank = _np(metrics.effective_rank_per_token(
        sigma,
        diagonal=diagonal,
        eps=eps,
        family=family,
    ))
    norm = Normalize(vmin=rank.min(), vmax=rank.max() + 1e-12)
    cmap = plt.cm.viridis
    fig, ax = plt.subplots(figsize=(5, 4.4))
    # A hundred overlapping ellipses read as one blob: outline a subset spread evenly over the
    # effective-rank order (both extremes included) and mark every token's mean.
    n_tokens = m.shape[0]
    shown = np.arange(n_tokens)
    if n_tokens > _ELLIPSE_MAX:
        by_rank = np.argsort(rank, kind="stable")
        shown = by_rank[np.unique(np.linspace(0, n_tokens - 1, _ELLIPSE_MAX).round().astype(int))]
    shapes = []
    for i in shown:
        if is_full:
            sub = s[i][np.ix_([a, b], [a, b])]
            w, v = np.linalg.eigh(0.5 * (sub + sub.T))
            ang = np.degrees(np.arctan2(v[1, 1], v[0, 1]))
            width, height = 2 * np.sqrt(np.clip(w[1], 0, None)), 2 * np.sqrt(np.clip(w[0], 0, None))
        else:
            ang, width, height = 0.0, 2 * np.sqrt(s[i, a]), 2 * np.sqrt(s[i, b])
        shapes.append((i, ang, width, height))
    # Largest first with a light fill and a rank-colored outline: overlapping opaque fills of a
    # hundred tokens otherwise merge into one blob that hides every small ellipse.
    for i, ang, width, height in sorted(shapes, key=lambda shape: -shape[2] * shape[3]):
        color = cmap(norm(rank[i]))
        ax.add_patch(Ellipse((m[i, a], m[i, b]), width=width, height=height, angle=ang,
                             facecolor="none", edgecolor=color, lw=0.9))
    ax.scatter(m[:, a], m[:, b], s=5, color="#9a9a9a", lw=0, zorder=2)
    ax.scatter(m[shown, a], m[shown, b], s=12, c=[cmap(norm(rank[i])) for i in shown],
               edgecolors="#26456E", linewidths=0.4, zorder=3)
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.8, label="effective rank")
    subtitle = (f"{shown.size} of {n_tokens} tokens outlined; gray dots: every mean"
                if shown.size < n_tokens else f"{n_tokens} tokens")
    ax.set(title=f"Belief covariance ellipses, dims {a} and {b}\n{subtitle}",
           xlabel=f"dim {a}", ylabel=f"dim {b}")
    ax.autoscale_view()
    return _save(fig, path)


@register_figure("holonomy_curvature")
def plot_holonomy_curvature(
    flat:      Dict,                     # holonomy_deviation_sampled output for the baseline run
    comparison: Optional[Dict] = None,   # ... for a second run to compare against

    *,
    curvature: Optional[object] = None,  # (N, N) curvature_field for a fixed anchor
    path:      Optional[str]    = None,
):
    r"""F10: holonomy / curvature with corrected sampling.

    Panel A: per-triangle holonomy violins -- the flat cocycle spikes at eps (roundoff of the
    coboundary construction, not a flatness certificate), a non-cocycle transport spreads to
    genuine curvature. Panel B: holonomy vs
    triangle span.
    Panel C (optional): the spatial curvature field for a fixed anchor.
    """
    ncol = 2 + (1 if curvature is not None else 0)
    fig, axes = plt.subplots(1, ncol, figsize=(4.4 * ncol, 3.7), squeeze=False)
    ax = axes[0][0]
    data = [np.clip(_np(flat["per_triple"]), 1e-12, None)]
    ticks = ["flat"]
    if comparison is not None:
        data.append(np.clip(_np(comparison["per_triple"]), 1e-12, None))
        ticks.append("comparison")
    ax.violinplot(data, showmeans=True)
    ax.set_yscale("log")
    ax.set(xticks=range(1, len(ticks) + 1), xticklabels=ticks,
           ylabel=r"$\|H_{ijk}-I\|_F$", title="Triangle holonomy")
    axb = axes[0][1]
    axb.scatter(_np(flat["span"]), np.clip(_np(flat["per_triple"]), 1e-12, None),
                s=10, color=_CB[0], alpha=0.5, label="flat")
    if comparison is not None:
        axb.scatter(_np(comparison["span"]), np.clip(_np(comparison["per_triple"]), 1e-12, None),
                    s=10, color=_CB[1], alpha=0.5, label="comparison")
        axb.legend(fontsize=8, frameon=False)
    axb.set_yscale("log")
    axb.set(xlabel="triangle span max|i-j|", ylabel=r"$\|H_{ijk}-I\|_F$", title="Curvature vs separation")
    if curvature is not None:
        # The field is the transport's own curvature, ~0 for the flat cocycle and genuinely
        # spread for a non-cocycle -- so the title must follow the run, not hard-code a mode name.
        field_regime = "comparison" if comparison is not None else "flat"
        im = axes[0][2].imshow(_np(curvature), cmap="magma", aspect="auto")
        fig.colorbar(im, ax=axes[0][2], shrink=0.8, label=r"$\|H-I\|_F$")
        axes[0][2].set(xlabel="j", ylabel="i", title=f"Curvature field ({field_regime})")
    fig.tight_layout()
    return _save(fig, path)


@register_figure("capacity_scaling")
def plot_capacity_scaling(
    scaling: Dict,                       # {axis_name: {x, bits_per_token, [lo, hi]}}

    *,
    path:    Optional[str] = None,
):
    r"""F11 Panel A: capacity scaling of validation bits/token vs each structural axis (K / heads /
    layers).

    Each sub-panel shares the bits/token axis; a bootstrap-over-validation-SEQUENCES band (NOT a
    cross-seed CI -- single-seed protocol) is drawn when ``lo`` / ``hi`` are supplied.
    """
    keys = list(scaling)
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 3.6), squeeze=False)
    for ax, key in zip(axes[0], keys):
        d = scaling[key]
        x, bits = _np(d["x"]), _np(d["bits_per_token"])
        order = np.argsort(x)
        ax.plot(x[order], bits[order], "o-", color=_CB[0], lw=1.8)
        if "lo" in d and "hi" in d:
            ax.fill_between(x[order], _np(d["lo"])[order], _np(d["hi"])[order], color=_CB[0], alpha=0.2)
        ax.set(xlabel=key, ylabel="validation bits/token", title=f"Scaling vs {key}")
    fig.suptitle("Capacity scaling (within-run bootstrap band)")
    fig.tight_layout()
    return _save(fig, path)


@register_figure("estep_capacity")
def plot_estep_capacity(
    n_e_steps:    object,                # (T,) E-step iteration counts
    bpc:          object,                # (T,) val BPC
    free_energy:  object,                # (T,) final-iterate objective

    *,
    n_params:          Optional[int] = None,
    wall_time:         Optional[object] = None,
    divergence_family: str = "renyi",
    evidence:          Optional[Mapping[str, object]] = None,
    path:              Optional[str] = None,
):
    r"""E-step inference depth versus validation BPC and the final-iterate objective.

    Parameters stay flat while inference depth changes. This displays the observed relationship;
    it does not presume monotonic improvement or convergence. An optional wall-time inset
    shows the compute cost.
    """
    x = _np(n_e_steps)
    objective_label, objective_units = _registered_divergence_axis(divergence_family)
    evidence = dict(evidence or {})
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.plot(x, _np(bpc), "o-", color=_CB[0], lw=2, label="val BPC")
    ax.set(xlabel="E-step iterations (n_e_steps)", ylabel="val BPC")
    ax2 = ax.twinx()
    ax2.plot(x, _np(free_energy), "s--", color=_CB[1], lw=2)
    ax2.set_ylabel(_objective_axis_text(
        f"final-iterate {objective_label}", objective_units), color=_CB[1])
    note = "inference-depth relationship" + (f"; params flat at {n_params:,}" if n_params else "")
    if evidence.get("fixed_point_evidence"):
        note += "; fixed point established"
    ax.set_title(f"E-step capacity ({note})")
    if wall_time is not None:
        ins = ax.inset_axes([0.6, 0.6, 0.36, 0.34])
        ins.plot(x, _np(wall_time), "o-", color="#888888", ms=3)
        ins.set(title="wall time (s)", xlabel="n_e_steps")
        ins.tick_params(labelsize=6)
    fig.tight_layout()
    return _save(fig, path)


@register_figure("pareto_frontier")
def plot_pareto_frontier(
    points: Dict,                        # {bits_per_token, n_params, [wall_time], [label]}

    *,
    path:   Optional[str] = None,
):
    r"""F11 Panel C: quality-vs-cost Pareto frontier of validation bits/token against parameters and
    wall time.

    Non-dominated cells (lower bits/token at lower cost) are connected into a stepwise frontier;
    dominated points are faded.
    """
    bits = _np(points["bits_per_token"])
    npar = _np(points["n_params"])
    have_time = "wall_time" in points
    ncol = 2 if have_time else 1
    fig, axes = plt.subplots(1, ncol, figsize=(5.0 * ncol, 4.0), squeeze=False)

    def _draw(ax, cost, xlabel, logx):
        order = np.argsort(cost)
        best = np.inf
        front = np.zeros(cost.size, dtype=bool)
        for k in order:
            if bits[k] < best:
                best = bits[k]; front[k] = True
        ax.scatter(cost[~front], bits[~front], s=24, color="#bbbbbb", label="dominated")
        ax.scatter(cost[front], bits[front], s=40, color=_CB[1], label="frontier")
        fo = order[np.isin(order, np.where(front)[0])]
        ax.plot(cost[fo], bits[fo], "-", color=_CB[1], lw=1.5)
        if logx:
            ax.set_xscale("log")
        ax.set(xlabel=xlabel, ylabel="validation bits/token")
        ax.legend(fontsize=8, frameon=False)

    _draw(axes[0][0], npar, "parameters", True)
    axes[0][0].set_title("Quality vs parameters")
    if have_time:
        _draw(axes[0][1], _np(points["wall_time"]), "wall time (s)", False)
        axes[0][1].set_title("Quality vs compute")
    fig.tight_layout()
    return _save(fig, path)


@register_figure("ablation_forest")
def plot_ablation_forest(
    rows: list,                          # [{label, delta, [lo], [hi]}] delta bits/token vs the full model

    *,
    n_seeds: int = 1,                    # distinct training seeds the sweep declared
    path: Optional[str] = None,
):
    r"""F12 Panel A: baseline-ladder ablation forest of delta bits/token from the full model.

    Each disabling ablation is a point with a paired sequence-cluster bootstrap interval over the
    shared validation stream, sorted by effect size; the zero line is the full model. The interval
    is within-run uncertainty over the evaluation set only; it omits training-seed variance, so
    unless the sweep declares at least 3 seeds the caption marks every arm as a single-seed
    estimate. The axis is delta bits/token (paired per-token nats divided by ln 2), NOT the
    character-corrected bits-per-character of the scaling figures.
    """
    rows = sorted(rows, key=lambda r: r["delta"])
    y = np.arange(len(rows))
    delta = np.array([r["delta"] for r in rows])
    lo = np.array([r.get("lo", r["delta"]) for r in rows])
    hi = np.array([r.get("hi", r["delta"]) for r in rows])
    fig, ax = plt.subplots(figsize=(6.4, 0.5 * len(rows) + 1.8))
    ax.errorbar(delta, y, xerr=[delta - lo, hi - delta], fmt="o", color=_CB[0],
                ecolor="#888888", capsize=3)
    ax.axvline(0.0, color="#444444", ls="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows])
    # The caption rides in the title block: the ablation driver adds its own bottom disclosure.
    caption = ("single-seed arms: within-run eval-set uncertainty only, no training-seed variance"
               if int(n_seeds) < 3 else
               f"{int(n_seeds)} seeds: per-cell within-run eval-set uncertainty")
    ax.set(xlabel=r"$\Delta$ bits/token vs full model")
    ax.set_title(f"Component ablations (paired sequence bootstrap)\n{caption}", fontsize=9)
    fig.tight_layout()
    return _save(fig, path)


@register_figure("lr_grid_heatmap")
def plot_lr_grid_heatmap(
    grid: Dict,                          # {x, y, z (len(y), len(x)), xlabel, ylabel, [baseline]}

    *,
    path: Optional[str] = None,
):
    r"""F12 Panel B: 2-D joint learning-rate sweep heatmap (val PPL), exposing ridge interactions.

    The basin minimum is starred and the baseline operating point marked, proving the tuned point is
    a joint minimum rather than a 1-D-slice artifact.
    """
    x, y, z = _np(grid["x"]), _np(grid["y"]), _np(grid["z"])
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.pcolormesh(x, y, z, cmap="viridis", shading="auto")
    fig.colorbar(im, ax=ax, shrink=0.85, label="val PPL")
    iy, ix = np.unravel_index(np.nanargmin(z), z.shape)
    ax.scatter([x[ix]], [y[iy]], marker="*", s=160, color="white", edgecolor="black", label="min")
    if grid.get("baseline") is not None:                   # a sweep may declare no baseline point
        bx, by = grid["baseline"]
        ax.scatter([bx], [by], marker="o", s=60, facecolor="none", edgecolor="red", label="baseline")
    ax.set(xlabel=grid.get("xlabel", "lr x"), ylabel=grid.get("ylabel", "lr y"),
           title="Joint LR sensitivity")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, path)


@register_figure("numerical_trust")
def plot_numerical_trust(
    guard:  Dict,                        # guard_saturation output (fractions)
    health: Dict,                        # numerical_health output (nan_* fractions, numel_* counts)
    causal: Optional[Dict] = None,       # causal_sanity output

    *,
    guard_denominators: Optional[Mapping[str, int]] = None,  # guard key -> entries examined
    population:         Optional[str]               = None,  # e.g. "test split, one sequence of 64 tokens"
    guard_scope:        Optional[str]               = None,  # e.g. "last layer": what the guard rows read
    leak_tol:           float                       = 1e-6,
    row_sum_tol:        float                       = 1e-5,
    path:               Optional[str]               = None,
):
    r"""F13: numerical trust as counts with denominators ("k of N"), one row per check.

    Guard rows count entries pinned at a numerical boundary: :math:`k=\mathrm{round}(fN)` from the
    saturation fraction :math:`f` of :func:`vfe3.metrics.guard_saturation` and the number :math:`N`
    of entries it examined (``guard_denominators``, from
    :func:`vfe3.viz.extract.guard_denominators`). Finiteness rows count non-finite entries of each
    intermediate (``nan_<x>`` fractions with ``numel_<x>`` counts from
    :func:`vfe3.viz.extract.numerical_health`). Causal rows count (layer, head) maps with future
    attention :math:`\max_{j>i}\beta_{ij}>` ``leak_tol`` and with row-sum error above
    ``row_sum_tol``. A missing denominator is printed as such rather than guessed. Nonzero counts
    are shaded. ``population`` names what the counts were measured on. The row groups may read
    different layers, so each names its own scope: ``guard_scope`` suffixes the guard rows, and the
    finiteness rows and the condition-number title carry "all L layers" when ``health["n_layers"]``
    is above one (``numerical_health`` pools every layer).
    """
    guard_denominators = dict(guard_denominators or {})
    guard_suffix = f" ({guard_scope})" if guard_scope else ""
    n_layers = health.get("n_layers")
    layer_suffix = (f" (all {int(n_layers)} layers)"
                    if isinstance(n_layers, (int, float)) and np.isfinite(float(n_layers))
                    and int(n_layers) > 1 else "")
    guard_names = {
        "sigma_floor_frac":       r"variance at the floor $\varepsilon$",
        "sigma_ceil_frac":        r"variance at the ceiling $\sigma_{\max}$",
        "energy_klmax_frac":      r"pair energy at $\mathrm{KL}_{\max}$ (all pairs)",
        "energy_klmax_frac_live": r"pair energy at $\mathrm{KL}_{\max}$ (causal pairs)",
        "selfdiv_klmax_frac":     r"self-divergence at $\mathrm{KL}_{\max}$",
    }
    symbols = {"mu": r"$\mu$", "sigma": r"$\Sigma$", "phi": r"$\phi$", "beta": r"$\beta$"}
    rows: List[List[str]] = []
    nonzero: List[bool] = []

    def _add(label: str, fraction: float, denominator: Optional[float]) -> None:
        if denominator is None or not np.isfinite(float(denominator)):
            rows.append([label, "n/a", "not recorded", f"{fraction:.3g}"])
        else:
            count = int(round(fraction * float(denominator)))
            rows.append([label, f"{count:,}", f"{int(denominator):,}", f"{fraction:.3g}"])
        nonzero.append(bool(fraction > 0.0))

    for key, label in guard_names.items():
        if key in guard:
            _add(label + guard_suffix, float(guard[key]), guard_denominators.get(key))
    for key in (k for k in health if k.startswith("nan_")):
        name = key.removeprefix("nan_")
        _add(f"non-finite {symbols.get(name, name)}{layer_suffix}", float(health[key]),
             health.get(f"numel_{name}"))
    if causal is not None:
        leak = _np(causal["future_leakage"]).astype(float).ravel()
        n_maps = int(leak.size)
        n_leaking = int((~(leak <= leak_tol)).sum())                 # NaN counts as leaking
        rows.append([f"(layer, head) maps with future attention > {_plain_tick(leak_tol)}",
                     f"{n_leaking:,}", f"{n_maps:,}", f"{n_leaking / max(n_maps, 1):.3g}"])
        nonzero.append(n_leaking > 0)
        if "row_sum_error" in causal:
            row_err = _np(causal["row_sum_error"]).astype(float).ravel()
            n_bad = int((~(row_err <= row_sum_tol)).sum())
            rows.append([rf"(layer, head) maps with $|{{\sum}}_j\beta_{{ij}}-1|$ > {_plain_tick(row_sum_tol)}",
                         f"{n_bad:,}", f"{int(row_err.size):,}",
                         f"{n_bad / max(int(row_err.size), 1):.3g}"])
            nonzero.append(n_bad > 0)
    if not rows:
        raise FigureSkipped("no guard, finiteness or causal counts to tabulate")
    with _compact_style():
        fig, ax = plt.subplots(figsize=(MAX_FIGURE_WIDTH_IN, 0.24 * (len(rows) + 1) + 0.75))
        ax.axis("off")
        table = ax.table(cellText=rows, colLabels=["check", "count k", "of N", "fraction k/N"],
                         colWidths=[0.58, 0.14, 0.14, 0.14], cellLoc="left", loc="upper center")
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.0, 1.25)
        for (row, col), cell in table.get_celld().items():
            cell.set_linewidth(0.4)
            # Text column flush left, count columns flush right, with a narrow inset.
            cell.PAD = 0.02 if col == 0 else 0.06
            cell.get_text().set_horizontalalignment("left" if col == 0 else "right")
            if row == 0:
                cell.set_text_props(fontweight="bold")
            elif nonzero[row - 1]:
                cell.set_facecolor("#FBE3C8")
        max_condition = health.get("max_condition")
        title = "Numerical trust: counts with denominators"
        if population:
            title += f" ({population})"
        if max_condition is not None and np.isfinite(float(max_condition)):
            title += f"\nlargest covariance condition number{layer_suffix}: {float(max_condition):.4g}"
        ax.set_title(title, fontsize=8)
        fig.tight_layout()
        return _save(fig, path)


# ===========================================================================
# Scaling-law figures (x-axis = number of parameters N; y = test CE in nats/token).
# Consume the cross-run aggregate produced by scaling_analysis.py: a list of
# per-point dicts {x_key: N, "ce_seeds": [per-seed test_ce], "route", "label"}.
# The shared power-law fit lives here (_fit_power_law) so both the figures and the
# analyzer tables use one implementation.
# ===========================================================================

def _fit_power_law(
    N:       object,                     # (P,) parameter counts (or tokens / FLOPs)
    L:       object,                     # (P,) loss at each point (test CE)

    *,
    weights:     object = None,          # (P,) WLS weights (e.g. (L/SEM)^2); None -> ordinary LS
    with_offset: bool   = False,         # True -> Chinchilla L = E + A N^-alpha (scipy); else log-log power law
) -> Dict[str, object]:
    r"""Fit ``L(N)``. Default: log-log (weighted) least squares ``L = A * N^{-alpha}`` (returns
    ``alpha = -slope``); ``with_offset=True`` fits the Chinchilla irreducible-loss form
    ``L = E + A * N^{-alpha}`` via ``scipy.optimize.curve_fit`` and silently falls back to the
    log-log fit if scipy is absent or the solve fails. Degenerate input (<2 distinct finite positive
    sizes) returns NaNs so the caller can skip the overlay; malformed weights raise ``ValueError``.
    """
    N = _np(N).astype(float).reshape(-1)
    L = _np(L).astype(float).reshape(-1)
    if N.size != L.size:
        raise ValueError(f"N and L must have equal lengths, got {N.size} and {L.size}")
    m = np.isfinite(N) & np.isfinite(L) & (N > 0) & (L > 0)
    w_all = None if weights is None else _np(weights).astype(float).reshape(-1)
    if w_all is not None and w_all.size != N.size:
        raise ValueError(f"weights must have length {N.size}, got {w_all.size}")
    N, L = N[m], L[m]
    w = None if w_all is None else w_all[m]
    if (w is not None and w.size > 0
            and (not np.all(np.isfinite(w)) or np.any(w < 0.0) or not np.any(w > 0.0))):
        raise ValueError("filtered weights must be finite, nonnegative, and contain a positive value")
    n_distinct_sizes = int(np.unique(N).size)
    out: Dict[str, object] = {"alpha": float("nan"), "A": float("nan"), "E": 0.0,
                              "r2": float("nan"), "n_points": int(N.size),
                              "n_distinct_sizes": n_distinct_sizes, "form": "power_law"}
    if with_offset and n_distinct_sizes < 4:
        out["form"] = "power_law_fallback_underdetermined"
    if N.size < 2 or n_distinct_sizes < 2:
        return out
    x, y = np.log(N), np.log(L)
    if with_offset and n_distinct_sizes >= 4:
        try:
            from scipy.optimize import curve_fit
            p0 = [max(0.0, float(L.min()) * 0.5), float(np.exp(np.mean(y))), 0.3]
            sig = None if w is None else (L / np.sqrt(np.clip(w, 1e-300, None)))   # m29: WLS (SEM=L/sqrt(w)); ignored when None
            popt, _ = curve_fit(lambda n, E, A, al: E + A * np.power(n, -al), N, L, p0=p0,
                                sigma=sig, absolute_sigma=False,
                                bounds=([0.0, 1e-12, 1e-3], [float(L.min()), np.inf, 3.0]),
                                maxfev=20000)
            E, A, al = (float(v) for v in popt)
            Lp = E + A * np.power(N, -al)
            ss_res = float(np.sum((L - Lp) ** 2))
            ss_tot = float(np.sum((L - L.mean()) ** 2))
            return {"alpha": al, "A": A, "E": E, "n_points": int(N.size),
                    "n_distinct_sizes": n_distinct_sizes, "form": "offset_power_law",
                    "r2": (1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"))}
        except Exception:
            out["form"] = "power_law_fallback_solver"
    import warnings
    with warnings.catch_warnings():                          # bootstrap resamples can repeat x's -> ill-conditioned
        warnings.simplefilter("ignore", np.exceptions.RankWarning)
        coef = np.polyfit(x, y, 1, w=(None if w is None else np.sqrt(np.clip(w, 0.0, None))))
    slope, intercept = float(coef[0]), float(coef[1])
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    out.update({"alpha": -slope, "A": float(np.exp(intercept)),
                "r2": (1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"))})
    return out


def _t95(n: int) -> float:
    r"""Two-sided 95% t-multiplier for ``n`` samples (``n-1`` dof); 0 for n<2, 1.96 if scipy absent."""
    if n < 2:
        return 0.0
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, n - 1))
    except Exception:
        return 1.96


def _scaling_point_stats(points: list, x_key: str = "n_params") -> Dict[str, Any]:
    r"""Per-point seed aggregation: returns x, mean, sem, ci95, n (one entry per point with >=1 finite
    seed), the flattened per-seed (seed_x, seed_y) cloud for the faint scatter, and the pooled
    pure-error variance of log CE over those points (``pooled_log_var``,
    :func:`_scaling_pooled_log_variance`)."""
    x, mean, sem, ci, ns, sx, sy, groups = [], [], [], [], [], [], [], []
    for p in points:
        seeds = [float(s) for s in p.get("ce_seeds", []) if np.isfinite(s)]
        try:
            xv = float(p.get(x_key, float("nan")))
        except (TypeError, ValueError):
            continue
        if not seeds or not np.isfinite(xv) or xv <= 0.0:
            continue
        arr = np.asarray(seeds, dtype=float)
        x.append(xv); mean.append(float(arr.mean())); ns.append(arr.size)
        s = float(arr.std(ddof=1)) / np.sqrt(arr.size) if arr.size > 1 else 0.0
        sem.append(s); ci.append(_t95(arr.size) * s)
        sx.extend([xv] * arr.size); sy.extend(seeds); groups.append(seeds)
    return {"x": np.asarray(x), "mean": np.asarray(mean), "sem": np.asarray(sem),
            "ci95": np.asarray(ci), "n": np.asarray(ns, dtype=int),
            "seed_x": np.asarray(sx), "seed_y": np.asarray(sy),
            "pooled_log_var": _scaling_pooled_log_variance(groups)}


def _scaling_pooled_log_variance(seed_groups: Sequence[Sequence[float]]) -> float:
    r"""Pooled pure-error variance of log CE over the seed groups of the fitted points.

    With :math:`y_{ij} = \log\mathrm{CE}_{ij}` (seed j of point i, :math:`n_i` seeds, m points, n runs),
    :math:`s^2 = \sum_{ij}(y_{ij} - \bar y_i)^2 / (n - m)`, the common-variance model of the analyzer's
    lack-of-fit test and offset-law bootstrap (``scaling_analysis``). NaN without a replicate seed
    or with zero spread.
    """
    logs = [np.log(np.asarray([c for c in g if np.isfinite(c) and c > 0.0], dtype=float)) for g in seed_groups]
    logs = [a for a in logs if a.size]
    df = sum(a.size for a in logs) - len(logs)
    ss = float(sum(np.sum((a - a.mean()) ** 2) for a in logs))
    return ss / df if df >= 1 and ss > 0.0 else float("nan")


def _scaling_sem_weights(
    mean: object,
    sem:  object,

    *,
    n_seeds:             object = None,          # (P,) seeds per point; with pooled_log_variance
    pooled_log_variance: Optional[float] = None, # s^2 of _scaling_pooled_log_variance
) -> np.ndarray:
    r"""Inverse-variance weights for a log-loss fit from point means and their SEMs.

    By the delta method, ``Var(log(mean)) ~= (SEM / mean)^2``, so the WLS weight is
    ``(mean / SEM)^2``. A point without its own SEM (a single seed, or seeds with zero spread)
    receives the pooled pure-error variance instead, weight ``n_i / s^2`` with ``s^2`` from
    :func:`_scaling_pooled_log_variance`, so it neither gains infinite leverage nor counts as a fitted
    size while carrying a vanishing weight next to the replicated points (audit 2026-10-10 #52).
    Without a pooled estimate (no replicated point) every point keeps unit weight.
    """
    mean_arr = _np(mean).astype(float).reshape(-1)
    sem_arr = _np(sem).astype(float).reshape(-1)
    if mean_arr.size != sem_arr.size:
        raise ValueError(f"mean and sem must have equal lengths, got {mean_arr.size} and {sem_arr.size}")
    weights = np.ones_like(mean_arr)
    positive = np.isfinite(mean_arr) & np.isfinite(sem_arr) & (mean_arr > 0.0) & (sem_arr > 0.0)
    weights[positive] = (mean_arr[positive] / sem_arr[positive]) ** 2
    if (n_seeds is not None and pooled_log_variance is not None
            and np.isfinite(pooled_log_variance) and pooled_log_variance > 0.0):
        n_arr = _np(n_seeds).astype(float).reshape(-1)
        if n_arr.size != mean_arr.size:
            raise ValueError(f"n_seeds must have length {mean_arr.size}, got {n_arr.size}")
        imputed = ~positive & np.isfinite(n_arr) & (n_arr >= 1.0)
        weights[imputed] = n_arr[imputed] / float(pooled_log_variance)
    return weights


def _scaling_route_label(route: str) -> str:
    if route == "blocks_K48_tied_2x":
        return f"{route} (tied structural ablation)"
    return route


def _scaling_fit_form_label(form: object) -> str:
    labels = {
        "offset_power_law": "offset power law",
        "power_law": "power law",
        "power_law_fallback_underdetermined": "power-law fallback (underdetermined)",
        "power_law_fallback_solver": "power-law fallback (solver failure)",
    }
    return labels.get(str(form), str(form).replace("_", " "))


@register_figure("scaling_law")
def plot_scaling_law(
    points: list,                        # [{n_params, ce_seeds:[...], route, label}, ...]

    *,
    x_key:       str  = "n_params",
    xlabel:      str  = "parameters N",
    with_offset: bool = False,
    title:       str  = "Scaling law (test CE vs parameters)",
    path:        Optional[str] = None,
):
    r"""Headline scaling figure: log-log ``test_ce`` vs ``N`` with per-seed points, cross-seed 95% CI
    error bars, the fitted power law overlaid, and a residual subpanel.

    The fit is a (SEM-weighted) log-log least squares ``L = A N^{-alpha}`` (or the Chinchilla
    ``E + A N^{-alpha}`` when ``with_offset``); the exponent ``alpha`` is annotated, and ``R^2`` only
    from :func:`~vfe3.metric_contracts.scaling_min_testable_sizes` distinct sizes of the fitted form (four for
    the two-parameter law, five for the offset law; fewer are labeled "not testable").
    The residual panel plots ``log(mean / fit)`` against ``log N`` -- systematic curvature there is
    the signature of a pre-asymptotic (not-yet-power-law) regime. The point count is annotated so a
    two-or-three-size fit is never read as more precise than it is."""
    st = _scaling_point_stats(points, x_key)
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(5.8, 5.4), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    if st["seed_x"].size:
        ax.scatter(st["seed_x"], st["seed_y"], s=14, color=_CB[0], alpha=0.22, label="per seed", zorder=2)
    if st["x"].size:
        ax.errorbar(st["x"], st["mean"], yerr=st["ci95"], fmt="o", color=_CB[1], ms=6,
                    capsize=3, lw=0, elinewidth=1.2, label="mean (95% CI)", zorder=3)
    fit = None
    if np.unique(st["x"]).size >= 2:
        w = _scaling_sem_weights(st["mean"], st["sem"], n_seeds=st["n"],
                                 pooled_log_variance=st["pooled_log_var"])
        fit = _fit_power_law(st["x"], st["mean"], weights=w, with_offset=with_offset)
        xx = np.geomspace(st["x"].min(), st["x"].max(), 100)
        yy = fit["E"] + fit["A"] * np.power(xx, -fit["alpha"])
        # Below scaling_min_testable_sizes distinct sizes the fitted law keeps <= 1 residual dof, so R^2
        # (exactly 1 at two sizes) tests nothing: annotate "not testable" instead of printing it. The gate is
        # the fitted form's (a with_offset fit that fell back to the plain law has two parameters), as in
        # scaling_analysis (audit 2026-10-16 F85).
        min_sizes = scaling_min_testable_sizes(fit["form"] == "offset_power_law")
        testable = int(fit["n_distinct_sizes"]) >= min_sizes
        lbl = (rf"fit $\alpha$={fit['alpha']:.3f}, "
               + (rf"$R^2$={fit['r2']:.3f}" if testable
                  else f"not testable (<{min_sizes} sizes)")
               + (f", E={fit['E']:.3f}" if fit["form"] == "offset_power_law" else "")
               + f", {_scaling_fit_form_label(fit['form'])}")
        ax.plot(xx, yy, "--", color="#444444", lw=1.6, label=lbl, zorder=4)
        yfit_pts = fit["E"] + fit["A"] * np.power(st["x"], -fit["alpha"])
        with np.errstate(divide="ignore", invalid="ignore"):
            resid = np.log(st["mean"] / yfit_pts)
        axr.axhline(0.0, color="#444444", ls="--", lw=1)
        axr.scatter(st["x"], resid, s=28, color=_CB[1])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set(ylabel="test CE (nats/token)", title=f"{title}  (P={int(st['x'].size)} sizes)")
    ax.legend(fontsize=8, frameon=False)
    axr.set_xscale("log")
    axr.set(xlabel=xlabel, ylabel=r"log(obs/fit)")
    fig.tight_layout()
    return _save(fig, path)


@register_figure("scaling_routes")
def plot_scaling_routes(
    points: list,                        # [{n_params, ce_seeds:[...], route, label}, ...]

    *,
    with_offset:      bool,

    x_key:            str = "n_params",
    xlabel:           str = "parameters N",
    title:            str = "Routes to N: does the frontier collapse?",

    weights_by_route: Optional[Mapping[str, np.ndarray]] = None,
    path:             Optional[str]                      = None,
):
    r"""Multi-route overlay: ``test_ce`` vs ``N`` with points colored AND markered by which knob grew
    ``N`` (embed_dim / gauge block size / depth / ...), each route's own power-law fit, and a dashed
    pooled fit. Routes that fall on one line share a frontier; a route offset above/below it (or a
    different slope) does not -- the visual companion to the analyzer's ANCOVA test. Marker shape
    duplicates the color so the routes survive grayscale printing."""
    routes = sorted({str(p.get("route", "?")) for p in points})
    pooled_log_var = _scaling_point_stats(points, x_key)["pooled_log_var"]   # one pure-error model for all routes
    route_stats = []
    for r in routes:
        st = _scaling_point_stats([p for p in points if str(p.get("route", "?")) == r], x_key)
        if not st["x"].size:
            continue
        if weights_by_route is None:
            weights = _scaling_sem_weights(st["mean"], st["sem"], n_seeds=st["n"],
                                           pooled_log_variance=pooled_log_var)
        else:
            if r not in weights_by_route:
                raise ValueError(f"weights_by_route is missing route {r!r}")
            weights = _np(weights_by_route[r]).astype(float).reshape(-1)
            if weights.size != st["x"].size:
                raise ValueError(
                    f"weights_by_route[{r!r}] has length {weights.size}; "
                    f"expected {st['x'].size} after point filtering"
                )
            if (not np.all(np.isfinite(weights)) or np.any(weights < 0.0)
                    or not np.any(weights > 0.0)):
                raise ValueError(
                    f"weights_by_route[{r!r}] must be finite, nonnegative, and contain a positive value"
                )
        route_stats.append((r, st, weights))

    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for i, (r, st, weights) in enumerate(route_stats):
        c, mk = _CB[i % len(_CB)], markers[i % len(markers)]
        ax.errorbar(st["x"], st["mean"], yerr=st["ci95"], fmt=mk, color=c, ms=6, capsize=2,
                    lw=0, elinewidth=1.0)
        if np.unique(st["x"]).size >= 2:
            fit = _fit_power_law(st["x"], st["mean"], weights=weights, with_offset=with_offset)
            xx = np.geomspace(st["x"].min(), st["x"].max(), 60)
            yy = fit["E"] + fit["A"] * np.power(xx, -fit["alpha"])
            ax.plot(xx, yy, color=c, lw=1.4,
                    label=rf"{_scaling_route_label(r)} ($\alpha$={fit['alpha']:.3f}; "
                          rf"{_scaling_fit_form_label(fit['form'])})")
        else:
            ax.plot([], [], marker=mk, color=c, lw=0, label=f"{_scaling_route_label(r)} (1 pt)")
    if route_stats:
        pooled_x = np.concatenate([st["x"] for _, st, _ in route_stats])
        pooled_mean = np.concatenate([st["mean"] for _, st, _ in route_stats])
        pooled_weights = np.concatenate([weights for _, _, weights in route_stats])
    else:
        pooled_x = pooled_mean = pooled_weights = np.array([], dtype=float)
    if np.unique(pooled_x).size >= 2:
        g = _fit_power_law(pooled_x, pooled_mean, weights=pooled_weights, with_offset=with_offset)
        xx = np.geomspace(pooled_x.min(), pooled_x.max(), 60)
        yy = g["E"] + g["A"] * np.power(xx, -g["alpha"])
        ax.plot(xx, yy, "--", color="#444444", lw=1.6,
                label=rf"pooled ($\alpha$={g['alpha']:.3f}; {_scaling_fit_form_label(g['form'])})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set(xlabel=xlabel, ylabel="test CE (nats/token)", title=title)
    ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    return _save(fig, path)


@register_figure("inference_capacity")
def plot_inference_capacity(
    series: Dict,                        # {knob_name: [{x, ce_seeds:[...]}, ...]} at FLAT params

    *,
    n_params: Optional[int] = None,
    title:    str = "Inference-compute capacity (flat N)",
    path:     Optional[str] = None,
):
    r"""Flat-``N`` inference-compute frontier: ``test_ce`` vs each inference depth knob (``n_e_steps``,
    ``n_layers``) at CONSTANT parameter count -- the architecture's unique capacity-from-inference
    axis. One panel per knob, x linear, mean +/- 95% CI over seeds. Kept SEPARATE from the ``L(N)``
    figures because depth/T add zero parameters and would make an ``L(N)`` curve spuriously
    multi-valued at fixed ``N``."""
    keys = [k for k in series if series[k]]
    if not keys:
        keys = list(series)
    fig, axes = plt.subplots(1, max(1, len(keys)), figsize=(4.6 * max(1, len(keys)), 3.8),
                             squeeze=False)
    for ax, k in zip(axes[0], keys):
        st = _scaling_point_stats(series[k], x_key="x")
        order = np.argsort(st["x"]) if st["x"].size else np.array([], dtype=int)
        if order.size:
            ax.errorbar(st["x"][order], st["mean"][order], yerr=st["ci95"][order], fmt="o-",
                        color=_CB[0], capsize=3, lw=1.6)
        ax.set(xlabel=k, ylabel="test CE (nats/token)", title=f"vs {k}")
    note = f"params flat at {n_params:,}" if n_params else "params flat"
    fig.suptitle(f"{title} ({note})")
    fig.tight_layout()
    return _save(fig, path)


# ===========================================================================
# Vocabulary next-token probability figures. Each takes a LIST of arms -- one
# arm (the single-run pipeline) renders one column; two arms (the cross-run
# driver, e.g. K70 vs K120) render side-by-side panels for a collapse contrast.
# Data come from vfe3.viz.extract.vocab_prediction_stats / decode_readout.
# ===========================================================================


def _tok_label(decode, tid) -> str:
    r"""A short printable label for vocab id ``tid`` (decoded text, or the id with no decoder)."""
    return token_label(int(tid), decode=decode)


def _distinct_token_labels(decode, ids: Sequence[int]) -> List[str]:
    r"""Labels of :func:`_tok_label` made distinct across ``ids``.

    Distinct tokens that differ only in whitespace (``"."`` and ``" ."``) share a label, so a label
    that repeats is rewritten with its spaces shown as ``·``; one still repeating gets its id.
    """
    from collections import Counter

    labels = [_tok_label(decode, t) for t in ids]
    counts = Counter(labels)
    out = []
    for tid, label in zip(ids, labels):
        if counts[label] > 1 and decode is not None:
            try:
                raw = decode([int(tid)])
            except Exception:
                raw = ""
            visible = raw.replace(" ", "\u00b7").replace("\n", "\n").replace("\t", "\t")
            if visible and visible.isprintable():
                label = visible if len(visible) <= 12 else visible[:11] + "\u2026"
        out.append(label)
    counts = Counter(out)
    return [f"{label} #{int(tid)}" if counts[label] > 1 else label for tid, label in zip(ids, out)]


@register_figure("vocab_probability_heatmap")
def plot_vocab_probability_heatmap(
    arms,                                # list of vocab_prediction_stats dicts (+ "label")

    *,
    decode: Optional[Callable] = None,
    cmap:   str = "magma",
    path:   Optional[str]      = None,
):
    r"""Seq x top-k heatmap of ``p(o_{n+1} | o_{<=n})``: rows are the most-probable tokens, columns
    the sequence positions, a green box marks the true next token. Bright horizontal bands (the same
    tokens lit at every position regardless of context) are the visual signature of prior collapse;
    a context-tracking model puts the bright cell on the boxed ground truth. One column per arm."""
    arms = list(arms)
    n = max(1, len(arms))
    R = max((int(_np(a["disp_probs"]).shape[0]) for a in arms), default=1)
    P = max((int(_np(a["disp_probs"]).shape[1]) for a in arms), default=1)
    fig, axes = plt.subplots(1, n, figsize=(max(4.5, 0.13 * P) * n, max(3.6, 0.17 * R)), squeeze=False)
    for ax, arm in zip(axes[0], arms):
        prob = _np(arm["disp_probs"])                            # (R, P)
        rows = _np(arm["row_ids"]).astype(int)
        truth = _np(arm["disp_truth_row"]).astype(int)
        pos = prob[prob > 0]
        vmax = float(prob.max()) if prob.size else 1.0
        vmin = float(pos.min()) if pos.size else vmax * 1e-4
        vmin = max(vmin, vmax * 1e-4, 1e-12)
        im = ax.imshow(prob, aspect="auto", cmap=cmap, norm=LogNorm(vmin=vmin, vmax=max(vmax, vmin * 1.1)))
        ax.set_yticks(range(prob.shape[0]))
        ax.set_yticklabels(_distinct_token_labels(decode, rows), fontsize=6)
        ax.set_xlabel("sequence position")
        for p in range(min(prob.shape[1], truth.shape[0])):
            r = int(truth[p])
            if r >= 0:
                ax.add_patch(plt.Rectangle((p - 0.5, r - 0.5), 1.0, 1.0, fill=False,
                                           edgecolor="#00E000", lw=0.7, zorder=5))
        ax.set_title(str(arm.get("label", "")))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="P(token | context)")
    fig.suptitle("Next-token vocabulary probabilities (rows = top predicted; green box = true next token)")
    fig.tight_layout()
    return _save(fig, path)


@register_figure("vocab_calibration")
def plot_vocab_calibration(
    arms,                                # list of vocab_prediction_stats dicts (+ "label")

    *,
    decode: Optional[Callable] = None,   # unused; kept for a uniform arm-figure signature
    path:   Optional[str]      = None,
):
    r"""Per-token mean predicted probability vs empirical next-token unigram frequency (log-log
    hexbin), with the ``y=x`` marginal-predictor reference. Mass hugging ``y=x`` together with a
    context-information gain ``H_uni - H_pred`` near zero means the model has collapsed to the
    marginal and is ignoring context. One column per arm."""
    arms = list(arms)
    n = max(1, len(arms))
    fig, axes = plt.subplots(1, n, figsize=(4.8 * n, 4.3), squeeze=False)
    for ax, arm in zip(axes[0], arms):
        uni = _np(arm["unigram"])
        mp = _np(arm["mean_pred_prob"])
        m = uni > 0
        x = np.log10(uni[m])
        y = np.log10(np.clip(mp[m], 1e-12, None))
        if x.size:
            hb = ax.hexbin(x, y, gridsize=40, bins="log", cmap="viridis", mincnt=1)
            fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.02, label="tokens (log)")
            # Over the x range only: drawn over the y range, the diagonal stretches the x axis far
            # past the data (predicted probabilities run decades below any unigram frequency).
            lo, hi = float(x.min()), float(x.max())
            ax.plot([lo, hi], [lo, hi], color=_CB[1], lw=1.2, ls="--", label="y = x (marginal predictor)")
            ax.legend(fontsize=7, frameon=False, loc="best")
        hp = float(arm.get("mean_pred_entropy", float("nan")))
        hu = float(arm.get("unigram_entropy", float("nan")))
        ax.set_xlabel(r"$\log_{10}$ empirical next-token frequency")
        ax.set_ylabel(r"$\log_{10}$ mean predicted probability")
        ax.set_title(rf"{arm.get('label', '')}: $H_{{\mathrm{{pred}}}}$ = {hp:.2f}, "
                     rf"$H_{{\mathrm{{uni}}}}$ = {hu:.2f}, gain = {hu - hp:.2f}")
    fig.suptitle("Vocabulary calibration\n(collapse: mass on y = x and gain near 0)")
    fig.tight_layout()
    return _save(fig, path)


@register_figure("vocab_confusion")
def plot_vocab_confusion(
    arms,                                # list of vocab_prediction_stats dicts (+ "label")

    *,
    decode:   Optional[Callable] = None,
    taxonomy: str = "function_content",
    path:     Optional[str]      = None,
):
    r"""Row-normalized next-token category confusion: true category (rows) vs argmax-predicted
    category (columns), bucketed by ``taxonomy`` (the 50257-token vocab is uncountable as a raw
    confusion matrix). Mass concentrating in a single predicted column -- everything predicted as
    one frequent category -- is collapse. Needs a decoder for the category bucketing. One column
    per arm."""
    if decode is None:
        raise ValueError("plot_vocab_confusion needs a token decoder for category bucketing")
    arms = list(arms)
    n = max(1, len(arms))
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 4.5), squeeze=False)
    for ax, arm in zip(axes[0], arms):
        true_ids = _np(arm["true_ids"]).astype(int)
        pred_ids = _np(arm["pred_ids"]).astype(int)
        tcat, names = _token_category_labels(true_ids, decode, taxonomy)
        pcat, _ = _token_category_labels(pred_ids, decode, taxonomy)
        C = len(names)
        M = np.zeros((C, C), dtype=float)
        np.add.at(M, (tcat.astype(int), pcat.astype(int)), 1.0)
        Mn = M / np.clip(M.sum(axis=1, keepdims=True), 1.0, None)
        im = ax.imshow(Mn, cmap="magma", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(C)); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(C)); ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("predicted category"); ax.set_ylabel("true next-token category")
        ax.set_title(str(arm.get("label", "")))
        for i in range(C):
            for j in range(C):
                ax.text(j, i, f"{Mn[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if Mn[i, j] < 0.6 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="P(predicted | true category)")
    taxonomy_name = {"function_content": "function/content"}.get(taxonomy, taxonomy.replace("_", " "))
    fig.suptitle(f"Next-token category confusion ({taxonomy_name}, row-normalized)")
    fig.tight_layout()
    return _save(fig, path)


@register_figure("decode_readout")
def plot_decode_readout(
    arms,                                # list of decode_readout dicts (+ "label")

    *,
    decode: Optional[Callable] = None,
    path:   Optional[str]      = None,
):
    r"""The linear-decode readout matrix ``W`` (``logits = mu_q @ W^T``): the top-norm vocabulary
    rows over the ``K`` latent dimensions, diverging colormap centered at zero. Structure here is
    the learned output embedding; a near-flat or low-rank readout under-uses the latent space. One
    column per arm."""
    arms = list(arms)
    n = max(1, len(arms))
    R = max((int(_np(a["weight"]).shape[0]) for a in arms), default=1)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, max(4.2, 0.06 * R)), squeeze=False)
    for ax, arm in zip(axes[0], arms):
        W = _np(arm["weight"])                                   # (R, K)
        rows = _np(arm["row_ids"]).astype(int)
        mag = float(np.abs(W).max()) or 1.0
        im = ax.imshow(W, cmap="coolwarm", vmin=-mag, vmax=mag, aspect="auto")
        step = max(1, W.shape[0] // 40)
        ax.set_yticks(range(0, W.shape[0], step))
        ax.set_yticklabels([_tok_label(decode, rows[i]) for i in range(0, W.shape[0], step)], fontsize=6)
        ax.set_xlabel("latent dimension k")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(str(arm.get("label", "")))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label=r"$W_{vk}$")
    fig.suptitle(r"Linear decode readout $W$ (top-norm vocabulary rows), $\mathrm{logits}=\mu_q W^{\top}$")
    fig.tight_layout()
    return _save(fig, path)


# ===========================================================================
# B1 / EXP-3  --  Sigma_q as calibrated Fisher uncertainty
# ===========================================================================

def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    r"""The headline :func:`vfe3.metrics.spearman_rho` (average ranks, paired-finite mask) on arrays."""
    import torch
    from vfe3.metrics import spearman_rho
    return spearman_rho(torch.as_tensor(np.asarray(x, float).ravel()),
                        torch.as_tensor(np.asarray(y, float).ravel()))


@register_figure("reliability_diagram")
def plot_reliability_diagram(
    reliability,                         # list of {conf, acc, frac} bins (run_artifacts._calibration_and_strata)

    *,
    ece:        Optional[float] = None,
    population: Optional[str]   = None,  # the sample the bins come from, e.g. "test split, n=8,320"
    path:       Optional[str]   = None,
):
    r"""B1/EXP-3 control: decode-calibration reliability diagram (accuracy vs confidence).

    The uncalibrated decode-softmax confidence baseline the Sigma_q-conditioned recalibration is
    measured against. Each bar is a confidence bin: height = empirical accuracy, the ``y=x`` diagonal
    is perfect calibration, and the gap (over/under-confidence) integrated and frac-weighted is the
    ECE annotated in the title. Consumes the bins ``run_artifacts`` already computes (conf / acc /
    frac) but never plotted. ``population`` is printed under the title, so bins from the report bank
    and bins from a ``research.json`` sample are not mistaken for one another."""
    rel = list(reliability or [])
    conf = np.array([b["conf"] for b in rel], float)
    acc = np.array([b["acc"] for b in rel], float)
    frac = np.array([b.get("frac", 0.0) for b in rel], float)
    if ece is None and rel:
        ece = float((frac * np.abs(acc - conf)).sum())
    with _compact_style():
        fig, ax = plt.subplots(figsize=(3.8, 3.7))
        ax.plot([0, 1], [0, 1], color=_CB[7], lw=1.0, ls="--", label="perfect calibration (y=x)")
        if conf.size:
            # Each bar is centered on its bin's mean confidence; narrow the edge bars so none
            # extends past [0, 1].
            width = np.minimum(0.9 / max(len(rel), 1), 2.0 * np.minimum(conf, 1.0 - conf))
            ax.bar(conf, acc, width=width, color=_CB[0], alpha=0.55,
                   edgecolor=_CB[0], label="empirical accuracy")
            ax.scatter(conf, acc, s=6.0 + 160.0 * frac, color=_CB[1], zorder=3,
                       label="bin (area ~ token frac)")
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="decode confidence (max softmax prob)",
               ylabel="empirical accuracy")
        title = "Decode reliability" + (f" (ECE = {ece:.3f})" if ece is not None else "")
        if population:
            title += f"\n{population}"
        ax.set_title(title)
        ax.legend(fontsize=6, frameon=False, loc="upper left")
        fig.tight_layout()
        return _save(fig, path)


def _sigma_trace_labels(frozen_covariance: bool) -> Dict[str, str]:
    r"""Axis / title wording for tr Sigma: belief uncertainty only when the covariance is live.

    On a route whose covariance is frozen at encode (``metrics.covariance_frozen_at_encode``) the
    trace is the input token's table variance tr Sigma(x_i), so the figures relabel it as such."""
    if frozen_covariance:
        return {"quantity": "input-token variance",
                "axis": r"input-token variance tr$\,\Sigma(x_i)$",
                "bin": r"tr$\,\Sigma(x_i)$"}
    return {"quantity": "belief uncertainty",
            "axis": r"belief uncertainty tr$(\Sigma_q)$",
            "bin": r"tr$(\Sigma_q)$"}


@register_figure("sigma_stratified_error")
def plot_sigma_stratified_error(
    bank: Dict,                          # belief_ce_bank: tr_sigma (M,), ce (M,)

    *,
    n_bins:  int = 10,
    n_boot:  int = 400,
    frozen_covariance: bool = False,     # covariance frozen at encode -> input-token variance labels
    path:    Optional[str] = None,
):
    r"""B1/EXP-3: held-out CE stratified by belief uncertainty tr(Sigma_q) (decile bins).

    The load-bearing calibration figure: if Sigma_q carries decode-time uncertainty, mean CE rises
    monotonically across increasing-tr(Sigma_q) bins. Tokens are split into ``n_bins`` equal-count
    quantile bins of tr(Sigma_q); each point is that bin's mean CE with a bootstrap 10-90 band
    (``n_boot`` resamples, fixed seed -> deterministic). A flat curve means the covariance channel is
    inert (see the CV>0.10 gate). With ``frozen_covariance`` the trace is the input token's table
    variance tr Sigma(x_i), so the figure is titled next-token CE by input-token variance: a
    type-level association, not a calibration of belief uncertainty."""
    labels = _sigma_trace_labels(frozen_covariance)
    title = (f"Next-token CE by {labels['quantity']}" if frozen_covariance
             else f"CE stratified by {labels['quantity']}")
    tr = _np(bank["tr_sigma"]).ravel()
    ce = _np(bank["ce"]).ravel()
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    if tr.size >= n_bins:
        edges = np.quantile(tr, np.linspace(0.0, 1.0, n_bins + 1))
        idx = np.clip(np.digitize(tr, edges[1:-1]), 0, n_bins - 1)   # interior edges -> n_bins bins
        rng = np.random.default_rng(0)
        xs, ms, los, his = [], [], [], []
        for b in range(n_bins):
            cb = ce[idx == b]
            if cb.size == 0:
                continue
            xs.append(float(tr[idx == b].mean()))
            ms.append(float(cb.mean()))
            boot = cb[rng.integers(0, cb.size, size=(n_boot, cb.size))].mean(axis=1)
            los.append(float(np.percentile(boot, 10)))
            his.append(float(np.percentile(boot, 90)))
        ax.plot(xs, ms, "o-", color=_CB[0], lw=1.8, label=f"mean CE per {labels['bin']} decile")
        ax.fill_between(xs, los, his, color=_CB[0], alpha=0.2, label="bootstrap 10-90 band")
        rho = _spearman(tr, ce)
        ax.set_title(rf"{title} (Spearman $\rho$ = {rho:+.3f})")
    else:
        ax.set_title(f"{title} (insufficient tokens)")
    ax.set(xlabel=labels["axis"], ylabel="cross-entropy (nats)")
    ax.xaxis.set_major_formatter(FuncFormatter(_count_label))
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, path)


@register_figure("sigma_ce_scatter")
def plot_sigma_ce_scatter(
    bank: Dict,                          # belief_ce_bank: tr_sigma (M,), ce (M,)

    *,
    frozen_covariance: bool = False,     # covariance frozen at encode -> input-token variance labels
    vocab_cv:   Optional[float] = None,  # CV of tr Sigma over the vocabulary types (frozen route)
    population: Optional[str]   = None,  # e.g. "test split, report bank"
    path: Optional[str] = None,
):
    r"""B1/EXP-3: per-token tr(Sigma_q) vs CE hexbin with the Spearman rho in the title.

    The raw joint distribution behind the stratified curve -- the headline statistic rho(tr Sigma_q,
    CE) is the rank correlation; a positive tilt means more-uncertain beliefs are harder tokens
    (the calibration channel carries signal). With ``frozen_covariance`` the trace is the input
    token's table variance tr Sigma(x_i): the axes say so, the title states that Sigma is frozen at
    encode, and the annotation gives the coefficient of variation

    .. math:: \mathrm{CV}=\frac{\mathrm{sd}(\mathrm{tr}\,\Sigma)}{|\mathrm{mean}(\mathrm{tr}\,\Sigma)|}

    over the plotted tokens (occurrence-weighted) and, when ``vocab_cv`` is given, over the
    vocabulary types. A CV of a few percent means the covariance is nearly constant, so the scatter
    shows token-type identity, not uncertainty."""
    labels = _sigma_trace_labels(frozen_covariance)
    tr = _np(bank["tr_sigma"]).astype(float).ravel()
    ce = _np(bank["ce"]).astype(float).ravel()
    with _compact_style():
        fig, ax = plt.subplots(figsize=(4.8, 3.6))
        if tr.size:
            # A log trace axis: a few tokens with a trace several times the median otherwise stretch a
            # linear axis until the bulk of the cloud fills its left fifth.
            positive = bool((tr > 0).all())
            hb = ax.hexbin(tr, ce, gridsize=40, bins="log", cmap="viridis", mincnt=1,
                           xscale="log" if positive else "linear")
            fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.02, label="tokens (log)")
            if not positive:
                ax.xaxis.set_major_formatter(FuncFormatter(_count_label))
        rho = _spearman(tr, ce)
        ax.set(xlabel=labels["axis"],
               ylabel=("next-token cross-entropy (nats)" if frozen_covariance
                       else "cross-entropy (nats)"))
        trace_symbol = (r"$\mathrm{tr}\,\Sigma(x_i)$" if frozen_covariance
                        else r"$\mathrm{tr}(\Sigma_q)$")
        title = trace_symbol + rf" vs CE (Spearman $\rho$ = {rho:+.3f}, n = {tr.size:,})"
        if frozen_covariance:
            title = r"$\Sigma$ frozen at encode: " + title
        ax.set_title(title)
        notes = []
        if tr.size >= 2 and abs(float(tr.mean())) > 0:
            occurrence_cv = float(tr.std(ddof=1) / abs(tr.mean()))
            notes.append(rf"CV(tr$\,\Sigma$) over these tokens = {occurrence_cv:.3f}")
        if vocab_cv is not None and np.isfinite(float(vocab_cv)):
            notes.append(rf"CV(tr$\,\Sigma$) over the vocabulary = {float(vocab_cv):.3f}")
        if frozen_covariance:
            notes.append("constant per input-token type: not a belief uncertainty")
        if population:
            notes.append(population)
        if notes:
            # Under the x label, not over the data: an in-axes box covers the high-CE hexbins.
            ax.annotate("\n".join(notes), xy=(0.0, 0.0), xycoords="axes fraction",
                        xytext=(0.0, -30.0), textcoords="offset points", va="top", ha="left",
                        fontsize=6, color="#444444")
        fig.tight_layout()
        return _save(fig, path)


# ===========================================================================
# F2 / EXP-7  --  prior-anchoring resists rank collapse
# ===========================================================================

@register_figure("rank_residual_by_depth")
def plot_rank_residual_by_depth(
    arms,                                # dict {label: (L,) r(X)} OR list of {"label", "rank_one_residual"}

    *,
    path: Optional[str] = None,
):
    r"""F2/EXP-7: Dong rank-one residual r(X) vs inference depth, one line per anchoring arm.

    r(X) -> 0 is rank collapse (every token mean identical); a no-anchor arm decays faster per layer
    than an alpha.KL(q||p)-anchored arm if prior-anchoring is the FFN-brake substitute. Each arm's
    log-linear decay rate b (slope of log r(X) vs layer) is annotated in the legend -- the per-arm
    decay-rate comparison the experiment decides on (absolute level is not the endpoint; the
    no-anchor control plateaus rather than collapsing to rank one)."""
    if isinstance(arms, dict):
        items = [(str(k), _np(v).ravel()) for k, v in arms.items()]
    else:
        items = [(str(a.get("label", i)), _np(a["rank_one_residual"]).ravel())
                 for i, a in enumerate(arms)]
    items = [(lab, r) for lab, r in items if r.size >= 1]
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for j, (lab, r) in enumerate(items):
        x = np.arange(1, r.size + 1)
        slope = ""
        if r.size >= 2 and float(np.min(r)) > 0:
            b = float(np.polyfit(np.arange(r.size), np.log(r), 1)[0])
            slope = f"  (b={b:+.3f})"
        ax.plot(x, r, "o-" if r.size > 1 else "o", color=_CB[j % len(_CB)], lw=1.6, label=f"{lab}{slope}")
    ax.set(xlabel="inference depth (layer)",
           ylabel=r"rank-one residual r(X) = $\|X-\mathbf{1}\bar x^T\|_F/\|X\|_F$")
    ax.set_title("Prior-anchoring vs rank collapse (Dong r(X) by depth)")
    if items:
        ax.set_xticks(np.arange(1, max(r.size for _, r in items) + 1))
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, path)


# ===========================================================================
# C2 / EXP-5  --  structural non-Neal-Hinton EM: final-iterate objective / CE relationship
# ===========================================================================

@register_figure("f_ce_decorrelation")
@register_figure("f_ce_relationship")
def plot_f_ce_relationship(
    arms,                                # list of {n_e_steps, final_f, ce, [label]} (one per n_e_steps cell)

    *,
    divergence_family: str = "renyi",
    path: Optional[str] = None,
):
    r"""Final-iterate target-blind objective per token versus held-out CE.

    Target blindness establishes structural separation from the prediction objective. It does not
    imply any correlation sign across a finite depth sweep; the displayed correlations are descriptive.
    Points are ordered and annotated by n_e_steps."""
    objective_label, objective_units = _registered_divergence_axis(divergence_family)
    pts = sorted(arms, key=lambda a: float(a["n_e_steps"]))
    ne = np.array([float(a["n_e_steps"]) for a in pts])
    ff = np.array([float(a["final_f"]) for a in pts])
    ce = np.array([float(a["ce"]) for a in pts])

    def _pear(a, b):
        return (float(np.corrcoef(a, b)[0, 1])
                if a.size >= 2 and a.std() > 0 and b.std() > 0 else float("nan"))
    r_fce, r_nef = _pear(ff, ce), _pear(ne, ff)
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    if ff.size:
        ax.plot(ff, ce, "-", color=_CB[7], lw=1.0, alpha=0.5, zorder=1)
        sc = ax.scatter(ff, ce, c=ne, cmap="viridis", s=90, zorder=3, edgecolor="k", linewidth=0.5)
        for a in pts:
            ax.annotate(f"T={int(float(a['n_e_steps']))}", (float(a["final_f"]), float(a["ce"])),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02, label="n_e_steps")
    ax.set(xlabel=_objective_axis_text(
               f"final-iterate {objective_label}", objective_units, per_token=True),
           ylabel="held-out CE (nats)")
    ax.set_title(rf"E-step objective / CE relationship  (Pearson$(F,CE)$={r_fce:+.3f}, Pearson$(T,F)$={r_nef:+.3f})")
    fig.tight_layout()
    return _save(fig, path)


plot_f_ce_decorrelation = plot_f_ce_relationship


# ===========================================================================
# D1 / EXP-8  --  pullback-group gauge M-step convergence
# ===========================================================================

@register_figure("wallclock_convergence")
def plot_wallclock_convergence(
    arms,                                # list of {label, val_ppl:[...], wall_clock_s:[...], step:[...]}

    *,
    target: Optional[float] = None,      # shared target PPL; None -> the worst arm's best (all reach it)
    path:   Optional[str]   = None,
):
    r"""D1/EXP-8: validation PPL vs cumulative wall time, one line per gauge M-step arm.

    The per-wall-clock convergence curve (the per-token pullback matrix_exp solve is the dominant
    added cost, so a per-STEP advantage can vanish per second). A shared target PPL (default: the
    worst arm's best, so every arm reaches it) is drawn; each arm's wall-time and step count TO that
    target are annotated in the legend -- the steps-to-target / wall-to-target convergence-speed
    readout. log-y so multiplicative gaps are legible."""
    arms = [a for a in arms if len(a.get("val_ppl", [])) >= 1 and len(a.get("wall_clock_s", [])) >= 1]
    if target is None and arms:
        mins = [min(a["val_ppl"]) for a in arms if a["val_ppl"]]
        target = max(mins) if mins else None
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for j, a in enumerate(arms):
        t = np.asarray(a["wall_clock_s"], float)
        y = np.asarray(a["val_ppl"], float)
        s = np.asarray(a.get("step", np.arange(t.size)), float)
        order = np.argsort(t)
        t, y, s = t[order], y[order], s[order]
        lab = str(a.get("label", j))
        if target is not None:
            hit = np.where(y <= target)[0]
            if hit.size:
                lab += f"  (→ {t[hit[0]]:.0f}s / {int(s[hit[0]])} steps)"
        ax.plot(t, y, "o-", color=_CB[j % len(_CB)], lw=1.6, ms=4, label=lab)
    if target is not None and np.isfinite(target):
        ax.axhline(target, color=_CB[7], ls="--", lw=1.0, alpha=0.6, label=f"target PPL={target:.2f}")
    ax.set_yscale("log")
    ax.set(xlabel="cumulative wall time (s)", ylabel="validation PPL (log)")
    ax.set_title("Gauge M-step convergence: val PPL vs wall-clock")
    if arms:
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, path)


# ===========================================================================
# Lower-priority experiment figures (auto-dispatched after their sweep / analysis)
#   A1/EXP-2 gauge transport · A2/EXP-9 equivariance drift · A3/EXP-10 CG coupling
#   H2/EXP-11 kappa dispersion · F1/EXP-6 muP K-stability · I1/EXP-1 noise band
# ===========================================================================

@register_figure("gauge_transport_bars")
def plot_gauge_transport_bars(
    cells,                               # list of {mode in {on,frozen,off}, depth, ppl, omega_dev}

    *,
    path: Optional[str] = None,
):
    r"""A1/EXP-2: gauge ON vs frozen-random vs OFF(Omega=I) validation PPL, grouped by depth L.

    The program's central causal probe: does trained GL(K) transport beat the exact Omega=I control?
    The OFF arm's max|Omega-I| (omega_dev) is annotated to certify the frame really collapsed to the
    identity to float eps. Predict ON < FROZEN <= OFF if transport carries the advantage."""
    cells = list(cells)
    modes = [("on", _CB[2]), ("frozen", _CB[4]), ("off", _CB[1])]
    depths = sorted({str(c["depth"]) for c in cells})

    def _v(mode, depth, key):
        m = [c for c in cells if c["mode"] == mode and str(c["depth"]) == depth]
        return float(m[0][key]) if m and m[0].get(key) is not None else float("nan")
    x = np.arange(len(depths)); w = 0.26
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for i, (mode, col) in enumerate(modes):
        ax.bar(x + (i - 1) * w, [_v(mode, d, "ppl") for d in depths], w, color=col, label=mode)
    for di, d in enumerate(depths):
        dev = _v("off", d, "omega_dev")
        if np.isfinite(dev):
            ax.annotate(rf"$|\Omega-I|$={dev:.0e}", (x[di] + w, _v("off", d, "ppl")),
                        textcoords="offset points", xytext=(0, 4), ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(depths)
    _zoom_bar_value_axis(                                  # padded window + headroom for the |Ω-I| labels
        ax, [_v(m, d, "ppl") for m, _ in modes for d in depths], hi_frac=0.20)
    ax.set(xlabel="depth", ylabel="validation PPL")
    ax.set_title("GL(K) gauge transport: ON vs frozen vs OFF (Ω=I)")
    ax.legend(fontsize=8, frameon=False, title="gauge")
    fig.tight_layout()
    return _save(fig, path)


@register_figure("ppl_equivariance_bars")
def plot_ppl_equivariance_bars(
    cells,                               # list of {label, ppl, resid}

    *,
    path: Optional[str] = None,
):
    r"""A3/EXP-10: per-arm validation PPL with the median gauge equivariance residual on a twin axis.

    The Clebsch-Gordan cross-irrep coupling is exactly equivariant for ANY learned path weights, so the
    residual stays ~eps on both arms (off/on) and the bar contrast is a clean PPL-delta at matched
    equivariance -- the capacity the only exactly-equivariant inequivalent-irrep channel buys."""
    cells = list(cells)
    labels = [str(c["label"]) for c in cells]
    x = np.arange(len(cells))
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ppls = [float(c["ppl"]) for c in cells]
    ax.bar(x, ppls, 0.5, color=_CB[0], label="val PPL")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    _zoom_bar_value_axis(ax, ppls, hi_frac=0.15)          # padded window so near-equal PPLs are legible
    for xi, v in zip(x, ppls):
        ax.annotate(f"{v:.2f}", xy=(xi, v), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, clip_on=False)
    ax.set_ylabel("validation PPL")
    resid = [float(c["resid"]) if np.isfinite(_as_f(c.get("resid"))) else np.nan for c in cells]
    if any(np.isfinite(resid)):
        axr = ax.twinx()
        axr.plot(x, resid, "s--", color=_CB[1], lw=1.4, label="equivariance residual")
        axr.set_ylabel("median equivariance residual")
        axr.set_yscale("log"); axr.grid(False)
    ax.set_title("Clebsch-Gordan coupling: PPL + equivariance residual")
    fig.legend(fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    return _save(fig, path)


@register_figure("kappa_dispersion")
def plot_kappa_dispersion(
    cells,                               # list of {label, dispersion, ppl}

    *,
    path: Optional[str] = None,
):
    r"""H2/EXP-11: validation PPL vs per-head temperature dispersion std(kappa_beta).

    Tied arms (uniform, and the geo-mean-tau confound control) sit at dispersion 0; the dispersed arms
    hold the arithmetic mean at 1.0 so the x-axis isolates per-head asymmetry. A separation between the
    tied baselines and the dispersed arms is the per-head-temperature effect (the geo-mean baseline
    controls for the mean-tau shift the dispersed arms carry)."""
    cells = list(cells)
    pts = sorted([(float(c["dispersion"]), float(c["ppl"]), str(c.get("label", ""))) for c in cells],
                 key=lambda t: t[0])
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=_CB[0], lw=1.6)
    for d, y, lab in pts:
        ax.annotate(lab, (d, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set(xlabel=r"per-head temperature dispersion std$(\kappa_\beta)$", ylabel="validation PPL")
    ax.set_title("Per-head temperature dispersion vs PPL")
    fig.tight_layout()
    return _save(fig, path)


@register_figure("kmup_stability")
def plot_kmup_stability(
    routes,                              # dict {route: [{embed_dim, ppl_mean, ppl_sem}...]}

    *,
    path: Optional[str] = None,
):
    r"""F1/EXP-6: inverse-K scaling width-fixed (grow_K) vs the muP Adam rate rule on the hidden and output
    matrices (grow_K_mup; not full muP) on a shared K=embed_dim axis.

    Test PPL vs K per route with cross-seed 95% CI bars (the small-sample t-quantile, not a fixed 1.96)
    and a requested offset power-law fit ``PPL = A K^{-alpha} + E`` overlaid when at least four
    distinct widths identify it. Smaller grids, including the live three-size ``grow_K`` route, show
    and label the realized log-log fallback without reporting an irreducible floor. The exponent
    ``b = -alpha`` is annotated per route. The grow_K_mup route is split into its matched /fixed and /mup arms upstream, so
    |b_fixed - b_muP| -- the width-stability readout -- is read directly off the two muP curves; a large
    gap means the headline exponent is partly optimization mis-tuning the muP rate rule removes."""
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    for j, (name, pts) in enumerate(routes.items()):
        pts = sorted(pts, key=lambda p: float(p["embed_dim"]))
        K = np.array([float(p["embed_dim"]) for p in pts], float)
        y = np.array([float(p["ppl_mean"]) for p in pts], float)
        ci = np.array([_t95(int(p.get("n", 2))) * float(p.get("ppl_sem", 0.0)) for p in pts], float)
        col = _CB[j % len(_CB)]
        ax.errorbar(K, y, yerr=ci, fmt="o", color=col, capsize=3, lw=1.4, label=name)
        if np.unique(K).size >= 2:
            fit = _fit_power_law(K, y, with_offset=True)
            a = fit.get("alpha", float("nan"))
            if np.isfinite(a):
                Kf = np.linspace(float(K.min()), float(K.max()), 100)
                ax.plot(Kf, fit["A"] * Kf ** (-a) + fit.get("E", 0.0), "-", color=col, lw=1.3, alpha=0.7)
                ax.plot([], [], " ",
                        label=f"   {name}: {_scaling_fit_form_label(fit['form'])}, b={-a:+.3f}")
    ax.set(xlabel="K (embed_dim)", ylabel="test PPL")
    ax.set_title("μP width-stability of the inverse-K exponent")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, path)


@register_figure("ppl_offset")
def plot_ppl_offset(
    points: list,                        # [{embed_dim, ppl_mean, [ppl_sem], [n_seeds]}, ...] pooled param points

    *,
    path:   Optional[str] = None,
):
    r"""Pooled test-PPL fit against width over all parameter points.

    The figure requests ``PPL = E + A K^{-alpha}`` when at least four distinct widths identify the
    three-parameter form and otherwise labels the realized log-log fallback without an ``E`` estimate.
    It pools every parameter route -- distinct from ``kmup_stability``, which splits
    the muP arms to read the width-stability contrast. Each point carries the across-seed mean PPL with
    a small-sample t-quantile CI when seeds are present; the realized fit and exponent ``b`` are overlaid."""
    pts = sorted([p for p in points
                  if np.isfinite(float(p.get("embed_dim", float("nan"))))
                  and np.isfinite(float(p.get("ppl_mean", float("nan")))) and float(p["ppl_mean"]) > 0],
                 key=lambda p: float(p["embed_dim"]))
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    if not pts:
        ax.text(0.5, 0.5, "no PPL-vs-embed_dim points", ha="center", va="center")
        ax.axis("off")
        return _save(fig, path)
    K = np.array([float(p["embed_dim"]) for p in pts], float)
    y = np.array([float(p["ppl_mean"]) for p in pts], float)
    ci = np.array([_t95(int(p.get("n_seeds", 2))) * float(p.get("ppl_sem", 0.0)) for p in pts], float)
    ax.errorbar(K, y, yerr=ci, fmt="o", color=_CB[0], capsize=3, lw=1.4, label="test PPL (across-seed)")
    if K.size >= 2 and float(K.max()) > float(K.min()):
        fit = _fit_power_law(K, y, with_offset=True)
        a = fit.get("alpha", float("nan"))
        if np.isfinite(a):
            Kf = np.linspace(float(K.min()), float(K.max()), 100)
            fit_label = f"{_scaling_fit_form_label(fit['form'])}  b={-a:+.3f}"
            if fit["form"] == "offset_power_law":
                fit_label += f"  E={fit['E']:.3f}"
            ax.plot(Kf, fit["A"] * Kf ** (-a) + fit.get("E", 0.0), "-", color=_CB[1], lw=1.6,
                    label=fit_label)
    ax.set(xlabel="K (embed_dim)", ylabel="test PPL", title="Pooled PPL fit vs width")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, path)


# =============================================================================
# Multi-seed digest: publication labels.
# Raw metrics.csv column / research scalar key -> publication-quality (mathtext) display label
# used for the across-seed digest figures (titles, axes, bar ticks). One source of truth so the
# helpers below render manuscript notation -- KL(q_i||p_i), sum_ij beta_ij KL(q_i||Omega_ij q_j),
# ... -- instead of the raw column names. Math is $...$ mathtext; pub_label() falls back to a
# humanized name for any key not listed. Notation matches the canonical free energy: q_i belief,
# p_i prior, Omega_ij transport, beta*_ij the canonical router softmax_j(log pi_ij - D_ij / tau) that
# weights the free-energy blocks (the model's attention only on a route whose router is that softmax; on
# the projected routes the blocks come from diagnose_canonical_free_energy), pi_ij prior attention, tau
# softmax temperature, alpha self-coupling weight with regularizer R(alpha), Sigma belief covariance
# (Sigma^{-1} precision / mean-block Fisher).
PUB_LABELS: Dict[str, str] = {
    # -- belief-channel free-energy blocks (raw per-token block energies) --
    "self_coupling":      r"$\sum_i [\alpha_i\,\mathrm{KL}(q_i\|p_i)+R(\alpha_i)]$",
    "self_divergence":    r"$\sum_i \mathrm{KL}(q_i\|p_i)$",
    "belief_coupling":    r"$\sum_{ij}\beta^*_{ij}\,\mathrm{KL}(q_i\|\Omega_{ij}q_j)$",
    "attention_entropy":  r"$\tau\sum_{ij}\beta^*_{ij}\log(\beta^*_{ij}/\pi_{ij})$",
    "total":              r"free energy $F$ per token (nats)",
    "canonical_beta_tv":  r"router distance $\frac{1}{2}\sum_j|\beta_{ij}-\beta^*_{ij}|$",
    "canonical_total_delta": r"$F_l(\mathrm{out})-F_l(\mathrm{in})$ (nats/token)",
    "inner_alignment_energy_total": r"inner alignment energy (nats/token)",
    "free_energy_total":  r"inner alignment energy (legacy column alias)",
    # -- attention / gauge / belief-geometry diagnostics --
    "attn_entropy":       r"attention row entropy $H(\beta)=-\sum_j\beta_{ij}\log\beta_{ij}$",
    "effective_rank":     r"belief effective rank $\mathrm{erank}(\Sigma)$",
    "belief_cond_median": r"belief conditioning $\mathrm{med}_i\,\kappa(\Sigma_i)$",
    "fisher_trace_mean":  r"Half Fisher trace $\mathrm{tr}(I_\mu)/2$",
    "val_fisher_trace_mean": r"Half Fisher trace $\mathrm{tr}(I_\mu)/2$",
    "grad_norm":          r"gradient norm $\|\nabla\|_2$",
    # -- losses / headline scalars --
    "train_ce":           "train cross-entropy (nats)",
    "val_ppl":            "validation perplexity",
    "val_bits_per_token": "validation bits per token",
    "val_bpc":            "validation bits per character",
    "generalization_gap": r"generalization gap $\mathrm{CE}_{\mathrm{val}}-\mathrm{CE}_{\mathrm{train}}$ (nats)",
    "test_ppl":           "test perplexity",
    "best_val_ppl":       "best validation perplexity",
    "test_ce":            "test cross-entropy (nats)",
    "test_bits_per_token": "test bits per token",
    "test_bpc":           "test bits per character",
    "test_ce_no_estep":   "test CE without E-step (nats)",
    "estep_capacity_gain": "E-step capacity gain (nats)",
    "wall_time_s":        "wall-clock time (s)",
    "ece":                "expected calibration error",
    "overall_ce":         "overall cross-entropy (nats)",
    "sigma_trace_cv":     r"$\mathrm{CV}_i[\mathrm{tr}\,\Sigma_i]$",
    "sigma_ce_spearman":  r"$\rho_s(\mathrm{tr}\,\Sigma,\ \mathrm{CE})$",
    "fd_gradient_worst_rel_error": "finite-diff worst rel. error",
    "corpus_freq_strata_ce.rare":     r"$\mathrm{CE}_{\mathrm{rare}}$ (nats)",
    "corpus_freq_strata_ce.mid":      r"$\mathrm{CE}_{\mathrm{mid}}$ (nats)",
    "corpus_freq_strata_ce.frequent": r"$\mathrm{CE}_{\mathrm{frequent}}$ (nats)",
    # -- geometry-health dashboard (gauge / SPD / Fisher / guard) --
    "gauge_invariant_spread":  r"frame class-function spread $\mathrm{std}_i\,I(\exp\phi_i)$",
    "phi_norm_mean":           r"gauge-frame norm $\langle\|\phi_i\|\rangle$",
    "gauge_head_phase_abs_mean": r"head frame phase $\langle|\arg\det_{\mathbb{C}}U_h|\rangle$",
    "belief_circularity_mean": r"belief circularity $\langle s_{\max}(R^{-1/2}PR^{-1/2})\rangle$",
    "belief_cond_p95":         r"belief conditioning $\kappa_{95}(\Sigma)$",
    "eff_rank_median":         r"belief effective rank $\mathrm{med}_i\,\mathrm{erank}(\Sigma_i)$",
    "fisher_trace_median":     r"Half Fisher trace $\mathrm{med}_i\mathrm{tr}(I_{\mu,i})/2$",
    "nonfinite_frac":          "non-finite fraction",
    "attn_entropy_min":        r"min row entropy $\min_{i,h}H(\beta)$",
    "attn_entropy_collapsed_heads": "collapsed-head count",
    # -- E-step inference quality --
    "estep_f_drop":            r"E-step $F_{\mathrm{end}}-F_{\mathrm{start}}$",
    "estep_f_nondecreasing_frac": "E-step nondecreasing fraction",
    "estep_r_mu_last":         r"E-step residual $r_\mu$ (last)",
    "estep_r_sigma_last":      r"E-step residual $r_\Sigma$ (last, SPD)",
    "estep_r_phi_last":        r"E-step residual $r_\phi$ (last)",
    # -- held-out validation sanity --
    "pos_loss_ratio":          r"positional loss ratio $\mathrm{CE}_{\mathrm{last}}/\mathrm{CE}_{\mathrm{first}}$",
    "val_future_leakage":      "causal future leakage (max)",
    "val_row_sum_error":       "attention row-sum error (max)",
    "val_pos_content_r2":      r"positional-content $R^2$",
    "val_head_redundancy_js":  "head redundancy (JS)",
    # -- pullback-group optimizer geometry --
    "phi_ridge_direction_cosine_mean":     r"ridge-direction cosine $\cos(v_\phi,g_\phi)$",
    "phi_pullback_damped_gen_cond_median": r"damped pullback condition $\mathrm{med}\ \kappa$",
    "phi_pullback_damped_gen_cond_max":    r"damped pullback condition $\max\ \kappa$",
    "phi_group_trust_scale_mean":          "accepted factor scale (mean)",
    "phi_group_trust_scale_min":           "accepted factor scale (min)",
    "phi_group_active_rows":               "active phi rows",
    "phi_group_chart_norm_max":            r"accepted chart norm $\max\|\phi\|_F$",
    "phi_group_backtracks_mean":           "factor-scale halvings (mean)",
    "phi_group_backtracks_max":            "factor-scale halvings (max)",
}


def pub_label(
    name: str,

    *,
    family: Optional[str] = None,
) -> str:
    r"""Publication-quality (mathtext) display label for a raw metric / column / scalar key.

    Returns the family-owned Fisher formula when ``family`` is supplied. Otherwise returns the
    :data:`PUB_LABELS` entry or a humanized fallback (underscores and dots -> spaces)."""
    if family is not None and name in {
        "fisher_trace_mean",
        "fisher_trace_median",
        "val_fisher_trace_mean",
    }:
        from vfe3.families.base import get_family
        return get_family(family).diagnostic_labels()["half_mean_fisher_trace"]
    return PUB_LABELS.get(name) or name.replace("_", " ").replace(".", " ")


@register_figure("ppl_noise_band")
def plot_ppl_noise_band(
    agg,                                 # aggregate_scalar output: {values, seeds, mean, sd}

    *,
    label: str  = "K=20 baseline",
    grid:  Optional[Dict] = None,        # optional {ablation label: single-seed ppl} overlay
    data_order_status: Optional[str] = None,   # multiseed data_order["status"], e.g. 'shared_explicit'
    path:  Optional[str]  = None,
):
    r"""I1/EXP-1: per-seed test PPL with the across-seed mean ± 1 SD and a 95% prediction band --
    the seed noise floor every ablation 'win' must clear.

    The outer band is the two-sided 95% prediction interval for ONE new run from the same seed
    distribution,

    .. math:: \bar{x} \pm t_{n-1,\,0.975}\, s \sqrt{1 + 1/n},

    since :math:`(X_{\mathrm{new}} - \bar{x}) / (s\sqrt{1 + 1/n}) \sim t_{n-1}`; a fixed ± 2 SD covers
    only 77.5% of new runs at n=3 and 86% at n=5. The band is the init+optimization spread; when the
    runs share one explicit data order (``data_order_status='shared_explicit'``) it omits data-order
    variance and is a lower bound on deployment variance, which the caption states. An optional
    ``grid`` overlays single-seed ablation cells so a referee can read directly whether a cell's
    margin exceeds the band."""
    vals = np.asarray(agg.get("values", []), float)
    mean, sd = float(agg.get("mean", float("nan"))), float(agg.get("sd", float("nan")))
    n = int(agg.get("n", vals.size) or vals.size)
    disp = pub_label(label)
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    if np.isfinite(mean):
        ax.axhline(mean, color=_CB[0], lw=1.6, label=f"mean={mean:.3f}")
        if np.isfinite(sd) and n >= 2:
            half_width = _t95(n) * sd * np.sqrt(1.0 + 1.0 / n)
            ax.axhspan(mean - half_width, mean + half_width, color=_CB[0], alpha=0.08,
                       label=rf"95% prediction ($t_{{{n - 1}}}$)")
            ax.axhspan(mean - sd, mean + sd, color=_CB[0], alpha=0.18, label="±1 SD")
    seeds = agg.get("seeds", list(range(vals.size))) or list(range(vals.size))
    xs = np.arange(vals.size)
    ax.scatter(xs, vals, color=_CB[1], zorder=3, label="per-seed value")
    ticks, ticklabels = list(xs), [str(s) for s in seeds[:vals.size]]
    if grid:
        gl = list(grid.items())
        gx = np.arange(vals.size, vals.size + len(gl))
        ax.scatter(gx, [float(v) for _, v in gl], color=_CB[3], marker="^", zorder=3, label="ablation cells")
        ticks += list(gx); ticklabels += [str(k) for k, _ in gl]
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels, rotation=45, ha="right", fontsize=7)
    ax.set(xlabel="seed / cell", ylabel=disp)
    caption = ("runs share one explicit data order: the spread omits data-order variance "
               "(a lower bound on deployment variance)"
               if data_order_status == "shared_explicit" else
               "data-order sharing not recorded" if data_order_status is None else
               "data order varies or is unspecified across runs")
    ax.set_title(f"Multi-seed variance floor (init + optimization spread)\n{caption}", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, path)


def plot_curve_band(
    steps: np.ndarray,                   # (S,) training-step grid
    mean:  np.ndarray,                   # (S,) across-seed mean per step
    sd:    np.ndarray,                   # (S,) across-seed SD (ddof=1; NaN where fewer than 2 seeds)

    *,
    label:  str  = "metric",
    ylabel: Optional[str] = None,        # y-axis; default -> pub_label(label) (publication math)
    n:      Optional[np.ndarray] = None, # (S,) seed count per step (for the title)
    logy:   bool = False,
    path:   Optional[str] = None,
):
    r"""Across-seed mean +/-1 SD (and +/-2 SD) ribbon of one training curve over steps.

    The band is the init+optimization spread (the per-run reseed shares the data order; a lower bound
    on deployment variance). Steps where only one seed reported (NaN SD, e.g. the sparse ``val_*``
    columns) draw the mean without a ribbon."""
    steps = np.asarray(steps, float)
    mean  = np.asarray(mean, float)
    sd    = np.asarray(sd, float)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    finite = np.isfinite(mean)
    ax.plot(steps[finite], mean[finite], color=_CB[0], lw=1.6, label="mean")
    band = finite & np.isfinite(sd)
    if band.any():
        ax.fill_between(steps[band], (mean - 2 * sd)[band], (mean + 2 * sd)[band],
                        color=_CB[0], alpha=0.10, label="$\\pm 2$ SD")
        ax.fill_between(steps[band], (mean - sd)[band], (mean + sd)[band],
                        color=_CB[0], alpha=0.22, label="$\\pm 1$ SD")
    if logy:
        ax.set_yscale("log")
    n_seeds = None
    if n is not None and np.any(np.isfinite(n)):
        n_seeds = int(np.nanmax(np.asarray(n, float)))
    ylab  = ylabel if ylabel is not None else pub_label(label)
    title = "across-seed mean $\\pm 1$ SD" + (f" ($n={n_seeds}$)" if n_seeds else "")
    ax.set(xlabel="training step", ylabel=ylab, title=title)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, path)


def plot_curve_band_grid(
    curves,                              # list of {steps, mean, sd, title, logy?}

    *,
    ncols: int = 3,
    path:  Optional[str] = None,
):
    r"""Multi-panel overview: one across-seed mean +/-1 SD ribbon per curve, shared step axis."""
    curves = list(curves)
    m = len(curves)
    ncols = max(1, min(ncols, m)) if m else 1
    nrows = int(np.ceil(m / ncols)) if m else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows), squeeze=False)
    for ax in axes.flat[m:]:
        ax.axis("off")
    for ax, c in zip(axes.flat, curves):
        steps = np.asarray(c["steps"], float)
        mean  = np.asarray(c["mean"], float)
        sd    = np.asarray(c["sd"], float)
        finite = np.isfinite(mean)
        ax.plot(steps[finite], mean[finite], color=_CB[0], lw=1.4)
        band = finite & np.isfinite(sd)
        if band.any():
            ax.fill_between(steps[band], (mean - sd)[band], (mean + sd)[band], color=_CB[0], alpha=0.22)
        if c.get("logy"):
            ax.set_yscale("log")
        ax.set_title(pub_label(c.get("title", "")), fontsize=9)
        ax.tick_params(labelsize=7)
    fig.supxlabel("training step", fontsize=9)
    fig.tight_layout()
    return _save(fig, path)


def plot_scalar_cv_summary(
    aggs,                                # {name: {mean, sd, cv, values}}

    *,
    path: Optional[str] = None,
):
    r"""Per-scalar across-seed coefficient of variation (SD / |mean|, %) as a sorted horizontal bar,
    with each seed's absolute normalized deviation (|value/mean - 1|) overlaid -- the seed-stability
    ranking a referee reads to know which headline numbers are robust and which ride seed noise."""
    items = [(k, v) for k, v in aggs.items()
             if v.get("cv") is not None and np.isfinite(float(v.get("cv", np.nan)))]
    items.sort(key=lambda kv: kv[1]["cv"])
    names = [k for k, _ in items]
    cvs   = [100.0 * float(v["cv"]) for _, v in items]
    fig, ax = plt.subplots(figsize=(7.4, max(2.4, 0.45 * len(names) + 1.0)))
    ys = np.arange(len(names))
    ax.barh(ys, cvs, color=_CB[5], alpha=0.7, zorder=2, label="across-seed CV")
    for y, (_, v) in zip(ys, items):
        mean = float(v.get("mean", np.nan))
        if np.isfinite(mean) and mean != 0.0:
            dev = [abs(100.0 * (float(val) / mean - 1.0)) for val in v.get("values", [])]
            ax.scatter(dev, [y] * len(dev), color=_CB[1], s=14, zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([pub_label(n) for n in names], fontsize=8)
    ax.set(xlabel="across-seed CV (% of mean); dots = |per-seed deviation|",
           title="Seed stability of headline scalars")
    fig.tight_layout()
    return _save(fig, path)


def plot_per_layer_band(
    per_layer,                           # {layer: {metric: {mean, sd, n, values}}}
    metric: str,

    *,
    path: Optional[str] = None,
):
    r"""Across-seed per-layer bars (mean) with +/-1 SD error bars for one ``metric``."""
    layers = sorted(per_layer.keys())
    means = [float(per_layer[l].get(metric, {}).get("mean", np.nan)) for l in layers]
    sds   = [float(per_layer[l].get(metric, {}).get("sd", np.nan)) for l in layers]
    sds   = [0.0 if not np.isfinite(s) else s for s in sds]
    fig, ax = plt.subplots(figsize=(max(4.0, 1.1 * len(layers) + 2.0), 4.0))
    xs = np.arange(len(layers))
    ax.bar(xs, means, yerr=sds, capsize=4, color=_CB[2], alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"layer {l}" for l in layers])
    ax.set(ylabel=pub_label(metric), title="per-layer mean across seeds ($\\pm 1$ SD)")
    fig.tight_layout()
    return _save(fig, path)


def _as_f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


@register_figure("pos_extrapolation")
def plot_pos_extrapolation(
    arms,                                # dict {label: [{n, ce}...]} OR list of {"label", "curve"}

    *,
    train_n: Optional[float] = None,
    path:    Optional[str]   = None,
):
    r"""H1/EXP-13: held-out CE vs eval sequence length N, one line per positional scheme.

    Offset attention priors (causal_alibi, a function of |i-j|) stay flat past the trained length; the
    absolute schemes (learned pos_phi table, RoPE) rise -- the extrapolation contrast. The train
    length is marked; points beyond it are pure extrapolation."""
    if isinstance(arms, dict):
        items = [(str(k), v) for k, v in arms.items()]
    else:
        items = [(str(a.get("label", i)), a.get("curve", [])) for i, a in enumerate(arms)]
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for j, (lab, curve) in enumerate(items):
        pts = sorted([(float(p["n"]), float(p["ce"])) for p in curve
                      if np.isfinite(_as_f(p.get("ce")))], key=lambda t: t[0])
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-",
                    color=_CB[j % len(_CB)], lw=1.6, ms=4, label=lab)
    if train_n is not None and np.isfinite(_as_f(train_n)):
        ax.axvline(float(train_n), color=_CB[7], ls="--", lw=1.0, alpha=0.6, label="train length")
    ax.set(xlabel="eval sequence length N", ylabel="held-out CE (nats)")
    ax.set_title("Positional extrapolation: offset vs absolute (CE vs N)")
    if items:
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, path)


@register_figure("renyi_saturation")
def plot_renyi_saturation(
    cells,                               # list of {alpha, attn_entropy, energy_klmax_frac}

    *,
    path: Optional[str] = None,
):
    r"""B2/EXP-12: attention entropy H(beta) and the kl_max energy-saturation fraction vs Renyi order.

    alpha<1 is mass-covering, alpha>1 mode-seeking; for alpha>1 the non-PD blend saturates the pairwise
    energy E_ij to kl_max with zero gradient (rising saturation fraction, right panel), which can drive
    a NON-MONOTONE H(beta)-vs-alpha tail (left panel) -- the saturation diagnostic explaining the
    entropy curve. alpha=1 (KL) is marked."""
    cells = sorted(cells, key=lambda c: float(c["alpha"]))
    a = [float(c["alpha"]) for c in cells]
    h = [float(c.get("attn_entropy", float("nan"))) for c in cells]
    s = [float(c.get("energy_klmax_frac", float("nan"))) for c in cells]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.2))
    ax1.plot(a, h, "o-", color=_CB[0], lw=1.8)
    ax1.axvline(1.0, color=_CB[7], ls="--", lw=1.0, alpha=0.5, label=r"$\alpha=1$ (KL)")
    ax1.set(xlabel=r"Rényi order $\alpha$", ylabel=r"attention entropy H($\beta$)")
    ax1.set_title("Attention diffuseness vs α")
    ax1.legend(fontsize=8, frameon=False)
    ax2.plot(a, s, "s-", color=_CB[1], lw=1.8)
    ax2.axvline(1.0, color=_CB[7], ls="--", lw=1.0, alpha=0.5)
    ax2.set(xlabel=r"Rényi order $\alpha$", ylabel="energy kl_max saturation fraction")
    ax2.set_title("Non-PD saturation vs α (the α>1 tail)")
    fig.suptitle("Rényi α-attention: entropy + saturation diagnostic (B2/EXP-12)")
    fig.tight_layout()
    return _save(fig, path)


@register_figure("mu_precond")
def plot_mu_precond(
    cells,                               # list of {precond in {fisher,raw}, n_e_steps, ppl}

    *,
    path: Optional[str] = None,
):
    r"""B3/EXP-14: validation PPL vs n_e_steps, Fisher natural-gradient vs raw-Euclidean mean step.

    The mu-arm ablation: nat_mu = Sigma*grad_mu (Fisher) vs the raw grad_mu, sigma retraction held
    fixed. One line per preconditioner; a gap that grows with n_e_steps means the mean-sector Fisher
    metric is load-bearing for the converged belief."""
    cells = list(cells)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for j, pre in enumerate(("fisher", "raw")):
        pts = sorted([(float(c["n_e_steps"]), float(c["ppl"])) for c in cells
                      if str(c["precond"]) == pre], key=lambda t: t[0])
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-",
                    color=_CB[j], lw=1.8, label=f"{pre} mean step")
    ax.set(xlabel="E-step iterations (n_e_steps)", ylabel="validation PPL")
    ax.set_title("Fisher nat-grad vs raw Euclidean E-step mean")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, path)


