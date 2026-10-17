r"""Orchestration and report writer for :mod:`vfe3.semantics`.

:func:`analyze_run` opens a trained run (:func:`~vfe3.semantics.run.load_semantic_run`), builds the
shared :class:`~vfe3.semantics.common.AnalysisContext`, and runs the sections in the order

1. ``phi`` -- gauge-fixed frame chart and frame semantics (:mod:`vfe3.semantics.frames`); it fills
   ``ctx.cache["canonical_frames"]`` for the ``phi_log`` view;
2. ``embeddings`` -- :func:`build_views` and the word-embedding battery of
   :mod:`vfe3.semantics.vectors` on every view, per-view figures/tables and the cross-view figures;
3. ``sigma`` -- covariance-table semantics (:mod:`vfe3.semantics.gaussians`);
4. ``contextual`` -- per-layer beliefs (:mod:`vfe3.semantics.contextual`, imported lazily).

Every section runs inside ``try/except``: a failure is recorded as a
:class:`~vfe3.semantics.common.SectionResult` with ``status="failed"`` and the traceback, and the
remaining sections still run. The run writes ``metrics.json`` (every section's metrics, the options,
the run facts, the resource coverage and the timings) and ``REPORT.md`` into the output directory.

Views (:func:`build_views`). Every view holds the rows of ``ctx.candidate_ids`` (the ``phi_log``
view drops frames without a real principal logarithm). The unit normalization ``D0`` of the
``mu_sigma_units`` and ``w2_chart`` views is :math:`\log d_{0,k} = -\tfrac12\,\mathrm{mean}_{v}
\log\sigma^2_{vk}` over all candidates, the same reference set the ``sigma`` section uses; the
``phi_log`` chart takes its ``D0`` and barycenter from the ``frame_ref_count`` most frequent
candidates, as the ``phi`` section does.

Presets (:data:`PRESETS`, option ``"preset"``). ``"full"`` keeps :data:`DEFAULT_OPTIONS`.
``"quick"`` is the lean per-seed battery of a multi-seed study: the views are the run's headline
decode view (``decode_W_causal`` for a linear decode, ``decode_natural_causal`` for a Gaussian KL
decode), ``phi_log``, ``mu_raw`` and ``xi_ref`` (:func:`preset_views`); no UMAP; the group-analogy
log rerank on 100 benchmark questions; geometry on 4,000 rows; a 16,384-token contextual bank and a
65,536-token occurrence budget. The occurrence budget serves the ambiguous words first, so the
token clouds are kept. Explicit options override the preset's values.

Timings. Measured for the full preset on the K160 run (``30.14_wikitext-103_K160_n=256_6-layer``,
K = 160, 6 layers, 20,072 candidates; 16 CPU threads; ``metrics.json`` timings): 1,644 s in total,
phi 257 s (canonical frames 59 s, group-analogy benchmark with 500 reranked questions 124 s),
embeddings 786 s over 9 views (per view battery + figures: phi_log 277 s, phi_raw 211 s,
decode_W_causal 57 s, mu_raw 35 s, xi_ref 34 s), sigma 29 s, contextual 570 s (bank 78 s, logit lens
28 s, occurrence forwards 444 s for 196,608 tokens). The quick preset on the same run is estimated,
not measured, from those per-step times: embeddings about 330-350 s (four views, no UMAP), phi about
150-230 s (the share of the 124 s rerank that 100 instead of 500 questions saves is not recorded
separately), sigma 29 s and contextual about 230 s on the CPU (bank 39 s, lens 28 s, occurrences about
150 s), about 12-14 minutes in total on 16 CPU threads. With ``device="cuda"`` the contextual forward
passes move to the GPU (the audit estimates 20-40 s for the full contextual budget); the frame chart
stays on the CPU in float64.
"""

from __future__ import annotations

import contextlib
import datetime
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from vfe3.semantics.common import (
    GAUSSIAN_BANK_DECODE_KINDS,
    AnalysisContext,
    SectionResult,
    VectorView,
    diagonal_redundancy_status,
    markdown_table,
    write_json,
)


DEFAULT_OPTIONS: Dict[str, Any] = {
    "preset":              "full",        # "full" | "quick" (PRESETS; explicit options override it)
    "min_count":           100,
    "max_candidates":      20000,
    "seed":                0,
    "device":              "cpu",
    "threads":             None,          # None -> cpu_count - 2
    "sections":            ("embeddings", "sigma", "phi", "contextual"),
    "umap":                True,
    "reference_position":  None,          # None -> min(max_seq_len, 256) // 2
    "frame_ref_count":     4096,
    "geometry_max_points": 8000,
    "contextual_tokens":   32768,
    "occurrence_tokens":   131072,
    "occurrence_split":    "train",
    "occurrence_per_word": 4,
    "context_len":         48,
    "views":               None,          # None -> every applicable view (subset of VIEW_NAMES)
    "headline_view":       None,          # None -> first of _HEADLINE_PRIORITY present
    "drop_unknown_config_fields": False,  # True: drop retired config toggles at load (load_semantic_run)
}

SECTIONS: Tuple[str, ...] = ("embeddings", "sigma", "phi", "contextual")
VIEW_NAMES: Tuple[str, ...] = (
    "mu_raw", "mu_sigma_units", "w2_chart", "xi_ref", "log_var", "phi_raw", "phi_log",
    "decode_W", "decode_W_causal", "decode_mu", "decode_natural_causal",
)
_HEADLINE_PRIORITY: Tuple[str, ...] = ("decode_W_causal", "decode_natural_causal", "decode_W",
                                       "decode_mu", "mu_raw")
_EXECUTION_ORDER: Tuple[str, ...] = ("phi", "embeddings", "sigma", "contextual")
_WHITEN_RTOL = 1e-10                   # covariance eigenvalues below rtol * max are dropped

# Option values each preset sets on top of DEFAULT_OPTIONS (explicit options win). The quick
# preset's views depend on the run's decode kind and are resolved after loading (preset_views).
PRESETS: Dict[str, Dict[str, Any]] = {
    "full":  {},
    "quick": {
        "umap":                   False,
        "frame_rerank_questions": 100,
        "geometry_max_points":    4000,
        "contextual_tokens":      16384,
        "occurrence_tokens":      65536,
    },
}
QUICK_VIEWS: Tuple[str, ...] = ("phi_log", "mu_raw", "xi_ref")   # plus the headline decode view
_HEADLINE_DECODE_VIEW: Dict[str, str] = {"linear": "decode_W_causal", "gaussian": "decode_natural_causal"}
CANDIDATE_IDS_FILE = "candidate_ids.npy"   # the run's candidate ids, frequency-descending (int64)


# ----------------------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------------------

def _log(message: str) -> None:
    r"""Progress line, made safe for the console encoding (cp1252 on Windows)."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    print(message.encode(encoding, errors="replace").decode(encoding, errors="replace"), flush=True)


def _get(obj: Any, *keys: Any) -> Any:
    for key in keys:
        if isinstance(obj, Mapping) and key in obj:
            obj = obj[key]
        elif isinstance(obj, (list, tuple)) and isinstance(key, int) and -len(obj) <= key < len(obj):
            obj = obj[key]
        else:
            return None
    return obj


def _num(value: Any) -> Optional[float]:
    r"""``float(value)`` for finite real numbers, else ``None``."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _f(value: Any, digits: int = 3) -> str:
    x = _num(value)
    return "–" if x is None else f"{x:.{digits}f}"


def _ok(result: Any) -> bool:
    return isinstance(result, Mapping) and result.get("status") not in ("failed", "skipped")


def _d_clause(cfg: Any) -> str:
    r"""What the global positive-diagonal rescaling (iii) is for this run, as a clause for view notes.

    Reads :func:`vfe3.semantics.common.diagonal_redundancy_status`: ``"exact"`` (measured for the
    configuration class), ``"broken"`` (a structural reason rules it out) or ``"assumed"``.
    """
    status, reason = diagonal_redundancy_status(cfg)
    if status == "exact":
        return f"an exact redundancy of this run ({reason})"
    if status == "broken":
        return f"not an exact symmetry of this run because {reason}, so D0 is a unit convention here"
    return f"assumed, not verified, to be a redundancy of this run ({reason})"


def _barycenter_status(chart: Any) -> Tuple[Optional[bool], Optional[float]]:
    r"""``(converged in every block, largest barycenter residual)`` from ``CanonicalFrames.diagnostics``.

    The residual is :math:`\max_b \|\tfrac1n\sum_a \log(m_b^{-1} U_{a,b})\|_F` at the returned
    barycenter (``barycenter_residual_max`` when the diagnostics carry it, else the maximum of
    ``blocks[*].barycenter.residual``); ``(None, None)`` when no chart diagnostics exist.
    """
    if not isinstance(chart, Mapping):
        return None, None
    converged = chart.get("barycenter_converged_all")
    residual = _num(chart.get("barycenter_residual_max"))
    if residual is None:
        values = [_num(_get(b, "barycenter", "residual")) for b in (chart.get("blocks") or [])]
        values = [v for v in values if v is not None]
        residual = max(values) if values else None
    return (None if converged is None else bool(converged)), residual


def _invariance_qualifier(chart: Any) -> str:
    r"""Empty when the frame barycenter converged; else the ``up to O(r)`` qualification.

    Equivariance :math:`\mathrm{mean}(gUh) = g\,\mathrm{mean}(U)\,h` holds at the fixed point, not at
    the best iterate, so a non-converged chart is invariant only up to the barycenter residual.
    """
    converged, residual = _barycenter_status(chart)
    if converged is not False:
        return ""
    if residual is None:
        return (" only approximately (the frame barycenter did not converge in every block and the best "
                "iterate was used)")
    return (f" only up to O(r), r = {residual:.1e} the largest barycenter residual (the frame barycenter did "
            "not converge in every block and the best iterate was used)")


def _reference_position(run: Any, ctx: AnalysisContext) -> int:
    position = ctx.options.get("reference_position")
    if position is None:
        return min(int(run.cfg.max_seq_len), 256) // 2
    return int(position)


def _causal_rows(X: np.ndarray) -> Tuple[np.ndarray, int]:
    r"""Center and whiten the rows: :math:`\bar x_v = \mathrm{Cov}(x)^{-1/2} (x_v - \bar x)`.

    Inner products of the returned rows are the causal inner product
    :math:`\langle x, y\rangle_C = x^\top \mathrm{Cov}(x)^{-1} y` of Park, Choe & Veitch (2024) with
    the covariance over the given rows. A singular covariance uses the pseudo-inverse square root
    on its range (eigenvalues above ``1e-10`` times the largest); the kept rank is returned.
    """
    X = np.asarray(X, dtype=np.float64)                                     # (n, D)
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = Xc.T @ Xc / max(X.shape[0] - 1, 1)                               # (D, D)
    w, V = np.linalg.eigh(cov)
    w_max = float(w.max()) if w.size else 0.0
    if w_max <= 0.0:
        raise ValueError("rows have zero covariance; the causal chart is undefined")
    keep = w > _WHITEN_RTOL * w_max
    Vk = V[:, keep]                                                         # (D, r)
    return Xc @ ((Vk / np.sqrt(w[keep])) @ Vk.T), int(keep.sum())


def _natural_parameters(run: Any, ids: np.ndarray) -> np.ndarray:
    r"""KL-decode natural parameters :math:`\eta_v = (m_v / t_v,\ -1/(2 t_v)) / \tau_\mathrm{eff}`, (n, 2K).

    For a diagonal belief :math:`q = N(\mu, \mathrm{diag}\,\Sigma)` the KL-decode logit is

    .. math:: -\frac{\mathrm{KL}(q\,\|\,N(m_v, \mathrm{diag}\, t_v))}{\tau_\mathrm{eff}} + \kappa\log\pi_v
              = \langle \lambda(q), \eta_v \rangle + c_v + c(q),
              \qquad \lambda(q) = (\mu,\ \mu^2 + \mathrm{diag}\,\Sigma),

    so :math:`\eta_v` is the decode "unembedding" row paired with the lifted belief. This holds
    only without the prior-bank head-evidence mixer, which weights coordinate :math:`k` of the KL by
    :math:`1 + c_{\delta,k}`; :func:`build_views` does not build the view when the mixer is on.
    """
    m = run.decode_mu(ids)                                                  # (n, K)
    t = np.exp(run.decode_log_var(ids))                                     # (n, K)
    tau = float(run.tau_eff())
    return np.concatenate([m / t, -0.5 / t], axis=1) / tau                  # (n, 2K)


# ----------------------------------------------------------------------------------------------
# Views
# ----------------------------------------------------------------------------------------------

def build_views(run: Any, ctx: AnalysisContext) -> Dict[str, VectorView]:
    r"""The token-indexed charts the embedding battery runs on, restricted to ``ctx.candidate_ids``.

    * ``mu_raw`` -- prior means :math:`\mu_v` (trained coordinates);
    * ``mu_sigma_units`` -- :math:`\mu_v d_0` with :math:`\log d_{0,k} = -\tfrac12\,\mathrm{mean}_v\,\log\sigma^2_{vk}`;
    * ``w2_chart`` (Euclidean) -- :math:`[\mu_v d_0,\ \sqrt{v_v}\, d_0]`, whose Euclidean distance is
      the 2-Wasserstein distance of the diagonal priors in D0 units;
    * ``xi_ref`` -- reference content :math:`\xi_{v,p} = e^{-Y_p} e^{-X_v}\mu_v`, per-block ZCA
      whitened (:func:`vfe3.semantics.frames.reference_content`), when frames are active;
    * ``log_var`` (Euclidean) -- :math:`\log\sigma^2_v` (for equal means the Fisher-Rao distance is
      :math:`\|\Delta\log\sigma^2\|/\sqrt 2`);
    * ``phi_raw`` -- raw algebra coordinates (trained trivialization), when frames are active;
    * ``phi_log`` -- flattened canonical logs :math:`L_v = \log\hat U_v` (``ctx.cache`` or
      :func:`vfe3.semantics.frames.canonical_frames`), rows without a real log excluded;
    * linear decode: ``decode_W`` rows and ``decode_W_causal`` (:func:`_causal_rows` of W);
      Gaussian decode: ``decode_mu`` and ``decode_natural_causal`` (:func:`_causal_rows` of
      :func:`_natural_parameters`).

    ``ctx.options["views"]`` (a subset of :data:`VIEW_NAMES`, or ``None`` for all) selects views.
    A view that does not apply is recorded in ``ctx.cache["view_notes"]``; a view whose
    construction raises is recorded there and its traceback in ``ctx.cache["view_errors"]``. An
    empty candidate set builds no view (noted). The status of the global positive-diagonal
    rescaling in the notes comes from :func:`vfe3.semantics.common.diagonal_redundancy_status`.
    """
    from vfe3.semantics import frames, gaussians

    requested = ctx.options.get("views")
    wanted = set(VIEW_NAMES) if requested is None else set(requested)
    unknown = wanted - set(VIEW_NAMES)
    if unknown:
        raise ValueError(f"unknown view names {sorted(unknown)}; known: {list(VIEW_NAMES)}")
    notes: List[str] = ctx.cache.setdefault("view_notes", [])
    errors: Dict[str, str] = ctx.cache.setdefault("view_errors", {})
    ids = np.asarray(ctx.candidate_ids, dtype=np.int64)
    views: Dict[str, VectorView] = {}
    if ids.size == 0:
        notes.append("no view built: the candidate set is empty (see coverage.candidates and the run notes)")
        return views
    d_clause = _d_clause(run.cfg)

    mu = run.mu(ids)                                                        # (n, K)
    lv = run.log_var(ids)                                                   # (n, K)
    mu_d0, log_d0 = gaussians.sigma_units(mu, lv)                           # (n, K), (K,)

    def attempt(name: str, make: Callable[[], Optional[VectorView]]) -> None:
        if name not in wanted:
            return
        try:
            view = make()
        except Exception as exc:                        # recorded: note + traceback, never silent
            errors[name] = traceback.format_exc()
            notes.append(f"view {name} failed to build: {type(exc).__name__}: {exc}")
            return
        if view is not None:
            views[name] = view

    attempt("mu_raw", lambda: VectorView(
        name="mu_raw", label="μ (raw)", ids=ids, X=mu, metric="cosine", group="mu",
        note=("Prior means μ_v in the trained coordinates. Cosine retrieval is not invariant under the "
              f"global positive-diagonal rescaling μ → Dμ, which is {d_clause}; analogy arithmetic "
              "a* − a + b is equivariant, the retrieval metric is not.")))
    attempt("mu_sigma_units", lambda: VectorView(
        name="mu_sigma_units", label="μ·d0 (D0 units)", ids=ids, X=mu_d0, metric="cosine", group="mu",
        note=("μ_v·d0 with log d0_k = −½ mean over the candidates of log σ²_vk (the Σ section's reference "
              "set). Invariant under μ → Dμ, log σ² → log σ² + 2 log D; that rescaling is "
              f"{d_clause}.")))
    attempt("w2_chart", lambda: VectorView(
        name="w2_chart", label="W2 chart (μ, √v)", ids=ids, X=gaussians.w2_chart(mu, lv, log_d0),
        metric="euclidean", group="sigma",
        note=("Euclidean chart [μ·d0, √v·d0]: the Euclidean distance is the 2-Wasserstein distance between "
              "the diagonal priors in D0 units, invariant under the global diagonal rescaling, which is "
              f"{d_clause}.")))

    def xi_ref() -> Optional[VectorView]:
        if not run.frames_active:
            notes.append("view xi_ref skipped: gauge frames are inactive for this run")
            return None
        p = _reference_position(run, ctx)
        return VectorView(
            name="xi_ref", label=f"ξ ref. content (p={p})", ids=ids,
            X=frames.reference_content(run, ids, p), metric="cosine", group="mu",
            note=(f"Reference content ξ_(v,p) = e^(−Y_p) e^(−X_v) μ_v at position p = {p}, ZCA-whitened per "
                  "frame block over the candidates: invariant under the split and diagonal redundancies, and "
                  "under the global right action h up to a per-block orthogonal map, which leaves cosines "
                  "unchanged."))

    attempt("xi_ref", xi_ref)

    def log_var_view() -> Optional[VectorView]:
        if float(np.ptp(lv, axis=0).max()) <= 0.0:
            notes.append("view log_var skipped: the log-variance table is constant over the candidates")
            return None
        return VectorView(
            name="log_var", label="log σ²", ids=ids, X=lv, metric="euclidean", group="sigma",
            note=("Log-variance rows log σ²_v (Euclidean). The global diagonal rescaling is a translation, so "
                  "distances are invariant; for equal means the Fisher–Rao distance is ‖Δ log σ²‖/√2."))

    attempt("log_var", log_var_view)

    def phi_raw() -> Optional[VectorView]:
        if not run.frames_active:
            notes.append("view phi_raw skipped: gauge frames are inactive for this run")
            return None
        return VectorView(
            name="phi_raw", label="φ raw coords", ids=ids, X=run.phi(ids), metric="cosine", group="phi",
            note=("Raw algebra coordinates φ_v in the trained trivialization. No function of a single raw φ_v "
                  "is invariant under the split redundancy e^(X_v) → e^(X_v) k; this view describes the "
                  "trained representative only."))

    attempt("phi_raw", phi_raw)

    def phi_log() -> Optional[VectorView]:
        if not run.frames_active:
            notes.append("view phi_log skipped: gauge frames are inactive for this run")
            return None
        cf = ctx.cache.get("canonical_frames")
        if cf is None or not np.array_equal(cf.ids, ids):
            n_ref = int(min(int(ctx.options.get("frame_ref_count", 4096)), ids.size))
            ref_ids = ids[:n_ref]
            cf = frames.canonical_frames(run, ids, ref_ids, run.log_var(ref_ids))
        converged, residual = _barycenter_status(cf.diagnostics)
        ctx.cache["phi_log_chart"] = {"barycenter_converged_all": converged, "barycenter_residual_max": residual}
        ids_ok, L = cf.log_chart()                                          # (n_ok,), (n_ok, sum d_b^2)
        n_bad = int(cf.n - ids_ok.size)
        if n_bad:
            notes.append(f"view phi_log: {n_bad} of {cf.n} frames have no real principal log and are excluded")
        return VectorView(
            name="phi_log", label="log Û (canonical)", ids=ids_ok, X=L, metric="cosine", group="phi",
            note=("Flattened principal logs L_v = log Û_v of the gauge-fixed frames Û_v = D0 e^(X_v) m^(−1) D0^(−1) "
                  f"(m: bi-invariant barycenter of the {cf.ref_ids.size} most frequent candidate frames, D0 from "
                  "their mean log-variance): invariant under the split redundancy and the global diagonal "
                  f"rescaling{_invariance_qualifier(cf.diagnostics)}; the rescaling is {d_clause}"
                  + (f"; {n_bad} rows without a real principal log are excluded." if n_bad else ".")))

    attempt("phi_log", phi_log)

    if run.decode_kind == "linear":
        W = run.decode_W(ids)                                               # (n, K)
        attempt("decode_W", lambda: VectorView(
            name="decode_W", label="decode W (raw)", ids=ids, X=W, metric="cosine", group="decode",
            note=("Linear-decode rows W_v (logits = μ Wᵀ + b + κ log π). Under the global diagonal rescaling "
                  f"W → W D^(−1), so raw cosine is not invariant; the rescaling is {d_clause}.")))

        def w_causal() -> VectorView:
            Z, rank = _causal_rows(W)
            return VectorView(
                name="decode_W_causal", label="decode W (causal)", ids=ids, X=Z, metric="cosine",
                group="decode",
                note=("W_v centered and whitened by Cov(W over the candidates)^(−1/2): the causal inner product "
                      f"of Park, Choe & Veitch (2024), rank {rank}/{W.shape[1]}. Invariant under any global "
                      "linear reparameterization of μ up to an orthogonal map."))

        attempt("decode_W_causal", w_causal)
    elif run.decode_kind in GAUSSIAN_BANK_DECODE_KINDS:
        readout = "KL-decode" if run.decode_kind == "gaussian" else "expected-likelihood decode"
        attempt("decode_mu", lambda: VectorView(
            name="decode_mu", label="decode m (raw)", ids=ids, X=run.decode_mu(ids), metric="cosine",
            group="decode",
            note=(f"{readout} means m_v (the untied decode bank when present). Raw cosine is not invariant under "
                  "the global diagonal rescaling m → Dm.")))

        def natural_causal() -> Optional[VectorView]:
            eta = _natural_parameters(run, ids)                             # (n, 2K)
            Z, rank = _causal_rows(eta)
            return VectorView(
                name="decode_natural_causal", label="decode η (causal)", ids=ids, X=Z, metric="cosine",
                group="decode",
                note=("Natural parameters η_v = (m_v/t_v, −1/(2 t_v))/τ_eff, in which the KL-decode logits are "
                      "linear in λ(q) = (μ, μ² + diag Σ); centered and whitened by Cov(η over the "
                      f"candidates)^(−1/2) (causal inner product), rank {rank}/{eta.shape[1]}. Invariant under "
                      "the global diagonal rescaling up to an orthogonal map."))

        if run.decode_kind == "gaussian":
            attempt("decode_natural_causal", natural_causal)
        elif "decode_natural_causal" in wanted:
            # The expected-likelihood logit -1/(2 tau_eff) sum_k [(mu - m_v)^2/(sigma + t_v) + log(sigma + t_v)]
            # couples the belief and bank variances, so no row eta_v pairs linearly with a lift of q.
            notes.append(f"view decode_natural_causal skipped: the {run.cfg.decode_mode!r} logits "
                         "log N(μ; m_v, diag(Σ + t_v))/τ_eff are not linear in λ(q) = (μ, μ² + diag Σ), so the "
                         "natural parameters η_v are not decode rows")
    else:
        notes.append(f"decode views skipped: decode_mode={run.cfg.decode_mode!r} is neither linear nor a "
                     "Gaussian decode-bank readout")
    if requested is not None:
        for name in VIEW_NAMES:
            if name in wanted and name not in views and name not in errors and not any(
                    name in n for n in notes):
                notes.append(f"view {name} not applicable to decode kind {run.decode_kind!r}")
    return views


# ----------------------------------------------------------------------------------------------
# Sections
# ----------------------------------------------------------------------------------------------

def _headline_view(views: Mapping[str, VectorView], ctx: AnalysisContext) -> Optional[str]:
    requested = ctx.options.get("headline_view")
    if requested is not None:
        if requested in views:
            return str(requested)
        ctx.cache.setdefault("view_notes", []).append(
            f"headline_view {requested!r} is not among the built views; using the default priority")
    for name in _HEADLINE_PRIORITY:
        if name in views:
            return name
    return next(iter(views), None)


def _embeddings_section(run: Any, ctx: AnalysisContext) -> SectionResult:
    r"""Views, the vector battery per view, per-view figures/tables and the cross-view figures."""
    from vfe3.semantics import vectors

    res = SectionResult(name="embeddings")
    timings: Dict[str, Any] = {}
    t = time.perf_counter()
    views = build_views(run, ctx)
    timings["build_views"] = time.perf_counter() - t
    # The phi_log view now holds what the report needs from the canonical frames; free their
    # float64 (n, d, d) blocks before the battery allocates its own copies.
    if ctx.cache.pop("canonical_frames", None) is not None:
        res.notes.append("ctx.cache['canonical_frames'] released after the phi_log view was built")
    _log(f"[embeddings] built {len(views)} views in {timings['build_views']:.1f} s: {', '.join(views)}")
    if not views:
        res.status = "skipped"
        res.notes.extend(ctx.cache.get("view_notes", []))
        res.notes.append("no view could be built")
        res.metrics = {"views": {}, "failed_views": dict(ctx.cache.get("view_errors", {}))}
        return res
    headline_view = _headline_view(views, ctx)
    ctx.options["headline_view"] = headline_view
    res.notes.extend(ctx.cache.get("view_notes", []))

    batteries: Dict[str, Dict[str, Any]] = {}
    failed: Dict[str, str] = dict(ctx.cache.get("view_errors", {}))
    view_seconds: Dict[str, Dict[str, float]] = {}
    for name, view in views.items():
        t = time.perf_counter()
        try:
            battery = vectors.run_vector_battery(view, ctx)
        except Exception as exc:                        # recorded: the other views still run
            failed[name] = traceback.format_exc()
            res.notes.append(f"view {name}: battery failed: {type(exc).__name__}: {exc}")
            _log(f"[embeddings]   {name}: battery FAILED ({type(exc).__name__}: {exc})")
            continue
        t_battery = time.perf_counter() - t
        batteries[name] = battery
        t = time.perf_counter()
        for label, make in (
                ("map", lambda: vectors.plot_view_embedding(view, ctx, battery)),
                ("offsets", lambda: vectors.plot_offsets(view, ctx, ctx.relations, battery=battery)),
                ("tables", lambda: vectors.write_view_tables(view, ctx, battery))):
            try:
                paths = make()
            except Exception as exc:                    # recorded in notes and the battery
                battery.setdefault("notes", []).append(f"{label} failed: {type(exc).__name__}: {exc}")
                battery.setdefault("output_errors", {})[label] = traceback.format_exc()
                res.notes.append(f"view {name}: {label} failed: {type(exc).__name__}: {exc}")
                continue
            if label == "tables":
                for path in paths:
                    res.tables[f"{name}/{Path(path).stem}"] = Path(path).read_text(encoding="utf-8")
            else:
                res.figures.extend(paths)
        view_seconds[name] = {"battery": t_battery, "outputs": time.perf_counter() - t}
        sub_failed = [k for k, v in battery.items() if isinstance(v, Mapping) and v.get("status") == "failed"]
        _log(f"[embeddings]   {name} ({view.n} x {view.dim}, {view.metric}): battery "
             f"{t_battery:.1f} s, figures/tables {view_seconds[name]['outputs']:.1f} s"
             + (f"; failed steps: {', '.join(sub_failed)}" if sub_failed else ""))
        for note in battery.get("notes", []):
            res.notes.append(note if str(note).startswith(f"{name}:") else f"view {name}: {note}")

    t = time.perf_counter()
    if batteries:
        try:
            res.figures.extend(vectors.plot_views_comparison(batteries, {n: views[n] for n in batteries}, ctx))
        except Exception as exc:                        # recorded; the batteries survive
            failed["comparison_figures"] = traceback.format_exc()
            res.notes.append(f"cross-view comparison figures failed: {type(exc).__name__}: {exc}")
    timings["comparison_figures"] = time.perf_counter() - t
    res.notes.extend(ctx.cache.get("vectors_notes", []))
    comparison_failures = list(ctx.cache.get("vectors_failed_steps", []))  # {"step": "compare.<name>", "error"}
    timings["views"] = view_seconds

    d_status, d_reason = diagonal_redundancy_status(run.cfg)
    res.metrics = {
        "views": {name: {"label": v.label, "metric": v.metric, "group": v.group, "n": v.n, "dim": v.dim,
                         "note": v.note} for name, v in views.items()},
        "headline_view": headline_view,
        "d0_reference": "all candidates (same as the sigma section)",
        "diagonal_redundancy_status": d_status,
        "diagonal_redundancy_reason": d_reason,
        "phi_log_chart": ctx.cache.get("phi_log_chart"),
        "batteries": batteries,
        "failed_views": failed,
        "failed_steps": comparison_failures,
        "timings": timings,
    }
    res.headline = _embeddings_headline(batteries, views)
    if not batteries:
        res.status = "failed"
        res.error = "every view's battery failed:\n" + "\n".join(failed.values())
    return res


def _analogy_accuracies(batteries: Mapping[str, Mapping[str, Any]]) -> Dict[str, Tuple[float, Optional[float]]]:
    r"""Per view, overall Google ``(3CosAdd accuracy, ONLY-B accuracy)`` (query words excluded)."""
    out: Dict[str, Tuple[float, Optional[float]]] = {}
    for v, b in batteries.items():
        acc = _num(_get(b, "analogy_benchmark", "overall", "methods", "3cosadd", "acc", "mean"))
        if acc is not None:
            out[v] = (acc, _num(_get(b, "analogy_benchmark", "overall", "methods", "only_b", "acc", "mean")))
    return out


def _macro_margin(bench: Mapping[str, Any]) -> str:
    r"""Macro-averaged (equal weight per Google category) 3CosAdd − ONLY-B, or "–" when not recorded."""
    a = _num(_get(bench, "overall", "macro", "methods", "3cosadd", "acc", "mean"))
    ob = _num(_get(bench, "overall", "macro", "methods", "only_b", "acc", "mean"))
    return f"{a - ob:+.3f}" if a is not None and ob is not None else "–"


def _stratified_null(purity: Optional[Mapping[str, Any]]) -> str:
    r"""The frequency-stratified purity null (labels permuted within log-count deciles) and its z."""
    strat = purity.get("frequency_stratified") if isinstance(purity, Mapping) else None
    if not isinstance(strat, Mapping) or _num(strat.get("null_mean")) is None:
        return "–"
    return f"{_f(strat['null_mean'])} (z = {_f(strat.get('z'), 1)})"


def _selection_tail(n: int) -> str:
    return f"maximum over {n} view{'s' if n != 1 else ''}, not adjusted for selection"


def _embeddings_headline(batteries: Mapping[str, Mapping[str, Any]],
                         views: Mapping[str, VectorView]) -> Dict[str, Any]:
    r"""Headline values; every cross-view maximum names its view and the number of views it is taken over.

    The analogy view is selected by the margin 3CosAdd − ONLY-B (Linzen 2016), and the
    highest-accuracy view is reported beside it.
    """
    head: Dict[str, Any] = {}
    acc = _analogy_accuracies(batteries)
    margin = {v: a - ob for v, (a, ob) in acc.items() if ob is not None}
    if margin:
        best = max(margin, key=margin.get)
        a, ob = acc[best]
        head["best Google 3CosAdd − ONLY-B margin (view)"] = (
            f"{margin[best]:.3f} ({views[best].label}: 3CosAdd {a:.3f} vs ONLY-B {ob:.3f}; "
            f"{_selection_tail(len(margin))})")
    if acc:
        best = max(acc, key=lambda v: acc[v][0])
        head["best Google 3CosAdd accuracy (view)"] = (
            f"{acc[best][0]:.3f} ({views[best].label}; ONLY-B {_f(acc[best][1])}; {_selection_tail(len(acc))})")
    for bench in ("simlex999", "wordsim353"):
        rho = {v: _num(_get(b, "similarity", bench, "rho")) for v, b in batteries.items()}
        rho = {v: r for v, r in rho.items() if r is not None}
        if rho:
            best = max(rho, key=rho.get)
            head[f"best {bench} Spearman ρ (view)"] = (
                f"{rho[best]:.3f} ({views[best].label}; {_selection_tail(len(rho))})")
    return head


def _run_section(name: str, fn: Callable[[], SectionResult]) -> SectionResult:
    r"""Run one section; any exception becomes a ``failed`` result carrying the traceback."""
    _log(f"[{name}] start")
    start = time.perf_counter()
    try:
        result = fn()
        if not isinstance(result, SectionResult):
            raise TypeError(f"section {name!r} returned {type(result).__name__}, not SectionResult")
    except Exception as exc:                            # recorded, the other sections continue
        result = SectionResult(name=name, status="failed", error=traceback.format_exc())
        result.notes.append(f"section failed: {type(exc).__name__}: {exc}")
    result.seconds = time.perf_counter() - start
    tail = "" if result.status != "failed" else f": {result.notes[-1] if result.notes else result.error}"
    _log(f"[{name}] {result.status} in {result.seconds:.1f} s{tail}")
    return result


def _sigma_section(run: Any, ctx: AnalysisContext) -> SectionResult:
    from vfe3.semantics import gaussians
    return gaussians.run_sigma_section(run, ctx)


def _phi_section(run: Any, ctx: AnalysisContext) -> SectionResult:
    if np.asarray(ctx.candidate_ids).size == 0:
        return SectionResult(name="phi", status="skipped",
                             notes=["the candidate set is empty, so there are no token frames to analyze"])
    from vfe3.semantics import frames
    return frames.run_phi_section(run, ctx)


def _contextual_section(run: Any, ctx: AnalysisContext) -> SectionResult:
    # Lazy import: a missing or broken contextual module fails only this section.
    from vfe3.semantics import contextual
    return contextual.run_contextual_section(run, ctx)


# ----------------------------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------------------------

def preset_views(preset: str, decode_kind: str) -> Optional[Tuple[str, ...]]:
    r"""The views a preset selects for a run of the given decode kind (``None``: every view).

    ``"quick"`` keeps the headline decode view of the decode kind (``decode_W_causal`` for a
    linear decode, ``decode_natural_causal`` for a Gaussian KL decode, none otherwise: the
    expected-likelihood decode has no reparameterization-invariant decode view) and
    :data:`QUICK_VIEWS`.
    """
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; known: {sorted(PRESETS)}")
    if preset != "quick":
        return None
    decode = _HEADLINE_DECODE_VIEW.get(decode_kind)
    return ((decode,) if decode is not None else ()) + QUICK_VIEWS


def _validate_device(device: Any) -> str:
    r"""The device string, after checking that torch parses it and that a CUDA device exists.

    A CUDA request on a machine without a usable CUDA device fails here, before any section
    runs, instead of partway through the contextual forwards.
    """
    try:
        parsed = torch.device(str(device))
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"device {device!r} is not a torch device: {exc}") from exc
    if parsed.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(f"device {device!r} requested but torch.cuda.is_available() is False")
        if parsed.index is not None and parsed.index >= torch.cuda.device_count():
            raise ValueError(f"device {device!r} requested but only {torch.cuda.device_count()} CUDA "
                             "device(s) exist")
    elif parsed.type != "cpu":
        raise ValueError(f"device {device!r}: only 'cpu' and 'cuda[:i]' are supported")
    return str(device)


def _resolve_options(options: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    r"""``DEFAULT_OPTIONS``, then the preset's values (:data:`PRESETS`), then the explicit ``options``."""
    user = dict(options or {})
    preset = user.get("preset", DEFAULT_OPTIONS["preset"])
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; known: {sorted(PRESETS)}")
    opts: Dict[str, Any] = {**DEFAULT_OPTIONS, **PRESETS[preset], **user}
    sections = tuple(opts["sections"])
    unknown = set(sections) - set(SECTIONS)
    if unknown:
        raise ValueError(f"unknown sections {sorted(unknown)}; known: {list(SECTIONS)}")
    opts["sections"] = sections
    if opts["views"] is not None:
        opts["views"] = tuple(opts["views"])
        bad = set(opts["views"]) - set(VIEW_NAMES)
        if bad:
            raise ValueError(f"unknown views {sorted(bad)}; known: {list(VIEW_NAMES)}")
    return opts


@contextlib.contextmanager
def _thread_limits(threads: int, notes: List[str]) -> Iterator[None]:
    r"""Pin torch intra-op threads and the BLAS/OpenMP pools to ``threads`` for the analysis."""
    previous = torch.get_num_threads()
    torch.set_num_threads(int(threads))
    try:
        try:
            from threadpoolctl import threadpool_limits
        except ImportError:
            notes.append("threadpoolctl unavailable: numpy/BLAS threads follow the environment variables")
            limiter: Any = contextlib.nullcontext()
        else:
            limiter = threadpool_limits(limits=int(threads))
        with limiter:
            yield
    finally:
        torch.set_num_threads(previous)


def _run_facts(run: Any, ctx: AnalysisContext) -> Dict[str, Any]:
    cfg = run.cfg
    d_status, d_reason = diagonal_redundancy_status(cfg)
    return {
        "run_dir": str(run.run_dir), "checkpoint": str(run.checkpoint_path), "dataset": run.dataset,
        "tokenizer": run.vocab.kind, "vocab_size": int(run.vocab_size), "K": int(run.K),
        "n_layers": int(run.n_layers), "n_heads": int(run.n_heads), "n_gen": int(run.n_gen),
        "block_dims": [int(s.stop - s.start) for s in run.block_slices],
        "family": getattr(cfg, "family", None), "sigma_init": getattr(cfg, "sigma_init", None),
        "gauge_group": getattr(cfg, "gauge_group", None),
        "gauge_transport": getattr(cfg, "gauge_transport", None),
        "transport_mode": "flat",                    # the only connection; the selector is retired
        "attention_mode": getattr(cfg, "attention_mode", None),
        "encode_mode": getattr(cfg, "encode_mode", None),
        "decode_mode": getattr(cfg, "decode_mode", None), "decode_kind": run.decode_kind,
        "decode_unigram_prior": getattr(cfg, "decode_unigram_prior", None),
        "pos_phi": getattr(cfg, "pos_phi", None), "max_seq_len": getattr(cfg, "max_seq_len", None),
        "norm_type_block": getattr(cfg, "norm_type_block", None),
        "layernorm_affine": getattr(cfg, "layernorm_affine", None),
        "amp_dtype_trained": getattr(cfg, "amp_dtype", None),
        "frames_active": bool(run.frames_active), "sigma_frozen": run.sigma_frozen,
        "phi_frozen": bool(run.phi_frozen), "counts_source": run.counts_source,
        "counts_diagnostics": dict(run.counts_diagnostics),
        "n_candidates": int(np.asarray(ctx.candidate_ids).size),
        "diagonal_redundancy_status": d_status, "diagonal_redundancy_reason": d_reason,
    }


def _json_ready(obj: Any, unknown: List[str], where: str = "") -> Any:
    r"""Map torch tensors, sets, slices and complex arrays to types :func:`common.to_jsonable` handles;
    anything else unknown becomes its ``repr`` and its location is appended to ``unknown``."""
    if isinstance(obj, Mapping):
        return {str(k): _json_ready(v, unknown, f"{where}/{k}") for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v, unknown, f"{where}[{i}]") for i, v in enumerate(obj)]
    if isinstance(obj, (set, frozenset)):
        return [_json_ready(v, unknown, where) for v in sorted(obj, key=str)]
    if isinstance(obj, torch.Tensor):
        return _json_ready(obj.detach().cpu().numpy(), unknown, where)
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in "biuf":
            return obj
        return _json_ready(obj.tolist(), unknown, where)
    if isinstance(obj, slice):
        return [obj.start, obj.stop]
    if obj is None or isinstance(obj, (str, bool, int, float, np.bool_, np.integer, np.floating, Path)):
        return obj
    unknown.append(f"{where or '/'}: {type(obj).__name__}")
    return repr(obj)


def _section_record(res: SectionResult, out_dir: Path) -> Dict[str, Any]:
    return {
        "status": res.status, "seconds": res.seconds, "notes": list(res.notes),
        "headline": dict(res.headline), "error": res.error,
        "figures": [_relative(p, out_dir) for p in res.figures],
        "tables": sorted(res.tables), "metrics": res.metrics,
    }


def _save_candidate_ids(path: Path, ids: np.ndarray) -> Path:
    r"""Atomically write the (n,) int64 candidate ids as ``.npy`` (the multi-seed stability reads them)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.npy")
    np.save(temporary, np.asarray(ids, dtype=np.int64).reshape(-1))
    os.replace(temporary, path)
    return path


def _relative(path: Any, root: Path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def analyze_run(
    run_dir:    "str | Path",

    *,
    out_dir:    "str | Path | None" = None,
    checkpoint: str = "best_model.pt",
    options:    Optional[Mapping[str, Any]] = None,
    vocab:      Any = None,
) -> Path:
    r"""Analyze one trained run and write ``metrics.json`` and ``REPORT.md``; returns the report path.

    ``out_dir`` defaults to ``<run_dir>/semantics``. ``options`` override :data:`DEFAULT_OPTIONS`
    and the preset (``options["preset"]``, :data:`PRESETS`); extra keys are passed through to the
    section modules, e.g. the ``frames.PHI_DEFAULTS`` budgets. The quick preset's views follow the
    run's decode kind (:func:`preset_views`) unless ``options`` names ``views`` explicitly.
    ``options["device"]`` (``"cpu"`` or ``"cuda[:i]"``, validated before loading) is
    ``ctx.device``: the contextual forward passes and the vector battery read it; the frame chart
    stays on the CPU in float64. ``vocab`` overrides the tokenizer vocabulary
    (``vocab.custom_vocab`` for non-standard runs). Sections run in the order phi, embeddings,
    sigma, contextual; each failure is recorded and the remaining sections still run. Besides
    ``metrics.json`` and ``REPORT.md`` the candidate ids are saved as :data:`CANDIDATE_IDS_FILE`
    (read by :mod:`vfe3.semantics.multiseed`). ``options["drop_unknown_config_fields"]`` reopens a
    run whose checkpoint config carries fields this code line no longer declares (retired toggles):
    they are dropped by :func:`~vfe3.semantics.run.load_semantic_run` and listed in ``run_notes``.
    """
    from vfe3.semantics.run import load_semantic_run, prepare_context

    opts = _resolve_options(options)
    opts["device"] = _validate_device(opts["device"])
    started = datetime.datetime.now()
    t_total = time.perf_counter()
    timings: Dict[str, Any] = {}

    t = time.perf_counter()
    run = load_semantic_run(run_dir, checkpoint=checkpoint, vocab=vocab,
                            drop_unknown_config_fields=bool(opts["drop_unknown_config_fields"]))
    timings["load"] = time.perf_counter() - t
    out = Path(out_dir) if out_dir is not None else run.run_dir / "semantics"
    _log(f"[semantics] run {run.run_dir} (K={run.K}, {run.n_layers} layers, {run.n_heads} heads, "
         f"decode {run.decode_kind}) loaded in {timings['load']:.1f} s; output {out}")
    if "views" not in dict(options or {}):
        views = preset_views(opts["preset"], run.decode_kind)
        if views is not None:
            opts["views"] = views
    _log(f"[semantics] preset {opts['preset']}; device {opts['device']}; views "
         + ("all applicable" if opts["views"] is None else ", ".join(opts["views"])))

    t = time.perf_counter()
    ctx = prepare_context(run, out, options=opts)
    timings["context"] = time.perf_counter() - t
    _save_candidate_ids(out / CANDIDATE_IDS_FILE, ctx.candidate_ids)
    _log(f"[semantics] context: {ctx.candidate_ids.size} candidates, {len(ctx.analogies)} Google questions, "
         f"{sum(len(v) for v in ctx.lexicon.values())} lexicon words; {ctx.threads} threads "
         f"({timings['context']:.1f} s)")

    runners: Dict[str, Callable[[Any, AnalysisContext], SectionResult]] = {
        "phi": _phi_section, "embeddings": _embeddings_section, "sigma": _sigma_section,
        "contextual": _contextual_section,
    }
    report_notes: List[str] = []
    results: Dict[str, SectionResult] = {}
    torch.manual_seed(ctx.seed)
    with _thread_limits(ctx.threads, report_notes):
        for name in _EXECUTION_ORDER:
            if name in opts["sections"]:
                results[name] = _run_section(name, lambda runner=runners[name]: runner(run, ctx))
    timings["sections"] = {name: res.seconds for name, res in results.items()}
    timings["total"] = time.perf_counter() - t_total

    unknown: List[str] = []
    payload = _json_ready({
        "schema": "vfe3.semantics.report/1",
        "created": started.isoformat(timespec="seconds"),
        "run": _run_facts(run, ctx),
        "options": ctx.options,
        "coverage": ctx.coverage,
        "run_notes": list(run.notes),
        "report_notes": report_notes,
        "timings": timings,
        "sections": {name: _section_record(res, out) for name, res in results.items()},
    }, unknown)
    if unknown:
        report_notes.append("metrics.json: non-JSON values stored as repr at " + "; ".join(unknown[:20]))
        payload["report_notes"] = list(report_notes)
    write_json(out / "metrics.json", payload)
    report_path = out / "REPORT.md"
    report_path.write_text(_render_report(run, ctx, results, timings, report_notes, started), encoding="utf-8")
    _log(f"[semantics] wrote {out / 'metrics.json'} and {report_path} ({timings['total']:.1f} s total)")
    return report_path


# ----------------------------------------------------------------------------------------------
# REPORT.md
# ----------------------------------------------------------------------------------------------

_FIGURE_CAPTIONS: Dict[str, str] = {
    "map": "PCA and UMAP maps of the semantic-lexicon words colored by field",
    "offsets": "relation offsets a → a* in a per-relation 2-D PCA",
    "analogy_categories": ("Google analogy 3CosAdd accuracy minus ONLY-B per category and view (cell text: "
                           "accuracy with ONLY-B in parentheses), with micro and macro rows"),
    "analogy_methods": "overall analogy accuracy per method and view",
    "similarity": "SimLex-999 / WordSim-353 Spearman ρ per view with bootstrap 95% CI",
    "fields": ("semantic-field kNN purity with the permutation null band (grey) and the frequency-stratified "
               "null band (tan), and silhouette per view"),
    "geometry": "effective rank, IsoScore, mean cosine and TwoNN intrinsic dimension per view",
    "directions": ("concept-direction cosines under the causal inner product, as the excess over a "
                   "frequency-matched random-target null (grey where |z| < 2)"),
    "scree": "PCA spectra of the centered rows per view",
    "frequency": "frequency confound: ridge probe of log count and PC1 / norm Spearman per view",
    "projections": "semantic projection onto antonym axes",
}
_COMPARE_CAPTIONS: Dict[str, str] = {
    **_FIGURE_CAPTIONS,
    "offsets": ("relation-offset consistency as the effect size (OCS − null)/(1 − null) against the re-pairing "
                "null per relation and view (dot: p < 0.05)"),
    "field_purity": "per-field kNN purity above chance, (P_f − c_f)/(1 − c_f), per view (n_f annotated)",
}
_SECTION_CAPTIONS: Dict[Tuple[str, str], str] = {
    ("sigma", "overview"): "covariance-table dispersion, log-variance histogram and initialization",
    ("sigma", "frequency_and_classes"): "log det Σ against log count and by token class",
    ("sigma", "fields"): "log det Σ by semantic field",
    ("sigma", "entailment"): "hypernymy from KL asymmetry against frequency and log-det baselines",
    ("sigma", "gaussian_similarity"): "Gaussian similarity scorers against cos(μ) and −‖Δμ‖",
    ("sigma", "routing_covariance"): "frame-induced routing covariance: Σ part and frame part of log det C",
    ("phi", "overview"): "frame norms, commutator ratios, BCH gaps and log-chart diagnostics",
    ("phi", "invariants_vs_frequency"): "gauge-fixed frame invariants against log count",
    ("phi", "group_analogies"): "group analogies Û_(a*) Û_a⁻¹ Û_b per category: chart, log rerank and ONLY-B",
    ("phi", "relation_transports"): "relation-transport consistency against the re-pairing null, eigen-phases",
    ("phi", "positional"): "positional frames: relative-position test and spectral log-norm heatmap",
    ("contextual", "logit_lens"): "logit lens per layer",
    ("contextual", "geometry"): "per-layer geometry of the contextual beliefs",
    ("contextual", "cka"): "layer × layer linear CKA",
    ("contextual", "contextuality"): "self-similarity and contextuality per layer",
    ("contextual", "layerwise_semantics"): "semantics of the per-layer distilled static views",
    ("contextual", "token_clouds"): "occurrence clouds of ambiguous words per layer",
    ("contextual", "sigma"): "contextual Σ per layer",
}

_SECTION_TITLES: Dict[str, str] = {
    "embeddings": "Embedding views", "sigma": "Covariance table Σ", "phi": "Gauge frames φ",
    "contextual": "Contextual beliefs",
}


def _caption(section: str, fig_rel: str) -> str:
    r"""One-line caption of a figure given its path relative to ``figures/``."""
    parts = fig_rel.split("/")
    stem = Path(fig_rel).stem
    if section == "embeddings":
        if len(parts) >= 3 and parts[1] == "compare":
            if stem.startswith("projections_"):
                return f"{_FIGURE_CAPTIONS['projections']} ({stem[len('projections_'):]})"
            return _COMPARE_CAPTIONS.get(stem, stem)
        return _FIGURE_CAPTIONS.get(stem, stem) + (f" ({parts[1]})" if len(parts) >= 3 else "")
    prefix = ""
    if section == "sigma" and stem.startswith("decode_"):
        prefix, stem = "decode bank: ", stem[len("decode_"):]
    return prefix + _SECTION_CAPTIONS.get((section, stem), stem.replace("_", " "))


def _md_cell(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


_NONE_WORD = re.compile(r"\bNone\b")


def _kv_table(items: Mapping[str, Any]) -> str:
    r"""Two-column table of headline values; a missing value (``None``, also inside a string) prints as –."""
    rows = [{"quantity": str(k), "value": _NONE_WORD.sub("–", v) if isinstance(v, str) else v}
            for k, v in items.items()]
    return markdown_table(rows, ["quantity", "value"])


def _google_total(ctx: AnalysisContext) -> Optional[int]:
    r"""Size of the full Google analogy benchmark (19544 questions), or ``None`` if it was unavailable."""
    total = _get(ctx.coverage, "google_analogies", "n_total")
    return int(total) if _num(total) is not None and total > 0 else None


def _covered_of_total(n_covered: Any, total: Optional[int]) -> str:
    r"""``covered/total (pct%)`` against the full benchmark size."""
    if total is None or _num(n_covered) is None:
        return "–" if _num(n_covered) is None else str(n_covered)
    return f"{int(n_covered)}/{total} ({100.0 * float(n_covered) / total:.1f}%)"


def _famous(battery: Mapping[str, Any], ctx: AnalysisContext,
            words: Tuple[str, str, str] = ("man", "woman", "king")) -> str:
    examples = battery.get("analogy_examples")
    if not isinstance(examples, list):
        return "–"
    for row in examples:
        if (ctx.display(row["a"]), ctx.display(row["a_star"]), ctx.display(row["b"])) != words:
            continue
        if not row.get("covered"):
            return "not in view"
        method = row["methods"].get("3cosadd")
        if method is None:
            return "–"
        return (", ".join(ctx.display(t) for t in method["topk"])
                + f" (rank of {ctx.display(row['b_star'])}: {method['rank']})")
    return "–"


def _acc_pair(group: Any) -> str:
    acc = _num(_get(group, "methods", "3cosadd", "acc", "mean"))
    only_b = _num(_get(group, "methods", "only_b", "acc", "mean"))
    if acc is None:
        return "–"
    return f"{acc:.3f} ({_f(only_b)})"


def _headline_tables(res: SectionResult, ctx: AnalysisContext) -> str:
    meta = res.metrics.get("views", {})
    batteries = res.metrics.get("batteries", {})
    if not batteries:
        return "_No view battery completed._\n"
    total = _google_total(ctx)
    acc = _analogy_accuracies(batteries)
    rows_a, rows_b = [], []
    for name, b in batteries.items():
        label = meta.get(name, {}).get("label", name)
        bench = b.get("analogy_benchmark")
        sim = b.get("similarity", {}) if isinstance(b.get("similarity"), Mapping) else {}

        def rho(bench_name: str) -> str:
            r = sim.get(bench_name)
            if not _ok(r) or _num(r.get("rho")) is None:
                return "–"
            return f"{r['rho']:.3f} (n={r['n']})"

        a, ob = acc.get(name, (None, None))
        rows_a.append({
            "view": label,
            "Google covered / benchmark": (_covered_of_total(bench.get("n_covered"), total or bench.get("n_total"))
                                           if _ok(bench) else "–"),
            "overall 3CosAdd − ONLY-B": f"{a - ob:+.3f}" if a is not None and ob is not None else "–",
            "macro 3CosAdd − ONLY-B": _macro_margin(bench) if _ok(bench) else "–",
            "semantic 3CosAdd (ONLY-B)": _acc_pair(bench.get("semantic")) if _ok(bench) else "–",
            "syntactic 3CosAdd (ONLY-B)": _acc_pair(bench.get("syntactic")) if _ok(bench) else "–",
            "honest top-1 = b": _f(_get(bench, "overall", "honest", "top1_is_b")) if _ok(bench) else "–",
            "man : woman :: king : ? (3CosAdd top-5)": _famous(b, ctx),
            "SimLex ρ": rho("simlex999"), "WordSim ρ": rho("wordsim353"),
        })
        fields = b.get("fields")
        geo = b.get("geometry") if _ok(b.get("geometry")) else {}
        purity = _get(fields, "purity") if _ok(fields) else None
        erank, dim = _num(_get(geo, "effective_rank")), _num(_get(geo, "dim"))
        probe = _get(geo, "frequency_probe")
        rows_b.append({
            "view": label, "D": meta.get(name, {}).get("dim", "–"),
            "field kNN purity@10 (null; z)": (f"{purity['purity']:.3f} ({_f(purity['null_mean'])}; "
                                             f"z = {_f(purity['z'], 1)})") if purity else "–",
            "frequency-stratified null (z)": _stratified_null(purity),
            "silhouette": _f(_get(fields, "silhouette", "silhouette")) if _ok(fields) else "–",
            "erank / D": f"{erank / dim:.3f}" if erank is not None and dim else "–",
            "IsoScore": _f(_get(geo, "isoscore")),
            "TwoNN": _f(_get(geo, "twonn", "mle"), 1),
            "mean cos raw / centered": f"{_f(_get(geo, 'anisotropy_raw'))} / {_f(_get(geo, 'anisotropy_centered'))}",
            "log-count probe R² (control)": (f"{_f(probe.get('r2_mean'))} ({_f(probe.get('control_r2_mean'))})"
                                             if _ok(probe) else "–"),
        })
    cols_a = list(rows_a[0].keys())
    cols_b = list(rows_b[0].keys())
    return ("Analogies (query words excluded; ONLY-B = nearest neighbor of b; coverage counts the questions "
            "whose four words resolve to candidate tokens in the view, out of the full Google set) and word "
            "similarity:\n\n"
            + markdown_table(rows_a, cols_a)
            + "\nSemantic fields (original space) and global geometry of the most frequent rows:\n\n"
            + markdown_table(rows_b, cols_b))


_PROBED_CLASS = ("projected-predictive block-GL runs with a coordinate MLP and no CG coupling, RoPE or "
                 "coordinate normalization")
_REDUNDANCIES = ("(i) the split e^(X_v) → e^(X_v) k, e^(Y_p) → k⁻¹ e^(Y_p) with one global gauge-group element "
                 "k; (ii) a global right action h on e^(Y_p) compensated in the relational operators; and (iii) a "
                 "global positive diagonal D acting as μ → Dμ, log σ² → log σ² + 2 log D, e^(X) → D e^(X)")


def _gauge_paragraph(run: Any, chart: Any = None) -> str:
    r"""The gauge-status paragraph of REPORT.md.

    The redundancies are called measured only when :func:`common.diagonal_redundancy_status` says
    the configuration is in the probed class (``"exact"``); otherwise (i) and (ii) are stated as
    assumed and (iii) as broken (with the reason) or assumed. ``chart`` (``CanonicalFrames``
    diagnostics) qualifies the chart invariance by the barycenter residual when it did not converge.
    """
    cfg = run.cfg
    status, reason = diagonal_redundancy_status(cfg)
    s = []
    if status == "exact":
        s.append("Three global redundancies were measured to leave the logits unchanged to fp32 round-off for the "
                 f"configuration class of this run ({_PROBED_CLASS}): {_REDUNDANCIES}.")
    else:
        text = ("Three global redundancies were measured to leave the logits unchanged to fp32 round-off only for "
                f"{_PROBED_CLASS}: {_REDUNDANCIES}. This run is outside that class, so (i) and (ii) are "
                "assumed, not verified, here, and (iii) is ")
        text += (f"not an exact symmetry here because {reason}." if status == "broken"
                 else f"assumed, not verified ({reason}).")
        s.append(text)
    s.append("Under (i) no statistic of a single raw φ_v is invariant except its per-head trace up to a global "
             "constant: the φ-raw view and raw frame norms, commutators and BCH gaps describe the trained "
             "trivialization, while the gauge-fixed chart Û_v = D0 e^(X_v) m⁻¹ D0⁻¹ (φ-log view, group analogies, "
             f"invariant scalars) is invariant under (i) and (iii){_invariance_qualifier(chart)}, and the pairwise "
             "transports e^(X_a) e^(−X_b) are invariant under (i) and change by a conjugation under (iii).")
    if status == "exact":
        s.append("Raw cosines of μ and of raw decode rows change under D, whereas the D0-unit views, the W2 chart, "
                 "the KL, Jeffreys and Fisher–Rao scorers, the causal (whitened) decode views and the reference "
                 "content ξ (up to a per-head orthogonal map) do not; analogy arithmetic a* − a + b is equivariant "
                 "under every global linear map, so only the retrieval metric has to be invariant.")
    elif status == "assumed":
        s.append("If (iii) is a redundancy of this run, raw cosines of μ and of raw decode rows change under it, "
                 "whereas the D0-unit views, the W2 chart, the KL, Jeffreys and Fisher–Rao scorers, the causal "
                 "(whitened) decode views and the reference content ξ (up to a per-head orthogonal map) do not; "
                 "analogy arithmetic a* − a + b is equivariant under every global linear map, so only the "
                 "retrieval metric has to be invariant.")
    else:
        s.append("Because D is not a symmetry here, D0 is a unit convention, not a gauge fixing; the causal "
                 "(whitened) decode views and the reference content ξ remain invariant under every global linear "
                 "reparameterization that the model admits.")
    s.append("Differences of log-variances and the ranking of log det Σ_v are invariant under D; raw variances "
             "are not.")
    extra = []
    if run.sigma_frozen:
        extra.append("Σ is frozen at encode on this route, so every layer's Σ equals the token prior and "
                     "contextual Σ analyses are vacuous")
    if run.phi_frozen:
        extra.append("φ is not updated in context, so the realized frame at position p is e^(X_v) e^(Y_p)")
    if run.frames_active:
        extra.append("with the flat transport Ω_ij = U_i U_j⁻¹ telescopes, so holonomy is "
                     "identically the identity and is not reported")
    if extra:
        text = "; ".join(extra)
        s.append(text[0].upper() + text[1:] + ".")
    return " ".join(s)


def _embeddings_summary(res: SectionResult, ctx: AnalysisContext) -> str:
    r"""Prose summary of the view batteries.

    The analogy view is the one with the largest margin 3CosAdd − ONLY-B; the highest-accuracy view is
    named too when it differs, and every view whose 3CosAdd does not exceed its ONLY-B baseline is
    listed. Coverage is stated against the full Google benchmark; cross-view maxima say over how
    many views they are taken and that they are not adjusted for selection.
    """
    batteries = res.metrics.get("batteries", {})
    meta = res.metrics.get("views", {})
    if not batteries:
        return ""

    def label(v: str) -> str:
        return meta.get(v, {}).get("label", v)

    s: List[str] = []
    acc = _analogy_accuracies(batteries)
    margin = {v: a - ob for v, (a, ob) in acc.items() if ob is not None}
    if margin:
        best = max(margin, key=margin.get)
        bench = batteries[best]["analogy_benchmark"]
        a = _get(bench, "overall", "methods", "3cosadd", "acc")
        total = _google_total(ctx) or bench.get("n_total")
        text = (f"On the {_covered_of_total(bench.get('n_covered'), total)} Google analogy questions whose four "
                f"words resolve to candidate tokens of the {label(best)} view, 3CosAdd with the query words excluded "
                f"reaches {a['mean']:.3f} [{_f(a.get('ci_lo'))}, {_f(a.get('ci_hi'))}] overall (semantic "
                f"{_f(_get(bench, 'semantic', 'methods', '3cosadd', 'acc', 'mean'))}, syntactic "
                f"{_f(_get(bench, 'syntactic', 'methods', '3cosadd', 'acc', 'mean'))}) against "
                f"{acc[best][1]:.3f} for the ONLY-B baseline that returns the nearest neighbor of b; its margin "
                f"{margin[best]:+.3f} over ONLY-B is the largest, a {_selection_tail(len(margin))}.")
        top_acc = max(acc, key=lambda v: acc[v][0])
        if top_acc != best:
            text += (f" The highest 3CosAdd accuracy, {acc[top_acc][0]:.3f} ({_selection_tail(len(acc))}), belongs "
                     f"to the {label(top_acc)} view, whose ONLY-B baseline is {_f(acc[top_acc][1])}.")
        not_above = [v for v in batteries if v in margin and margin[v] <= 0.0]
        if "mu_raw" in acc and "mu_raw" not in (best, top_acc, *not_above):
            text += (f" The raw prior means reach {acc['mu_raw'][0]:.3f} against ONLY-B "
                     f"{_f(acc['mu_raw'][1])}.")
        s.append(text)
        if not_above:
            s.append("3CosAdd does not exceed its ONLY-B baseline for the "
                     + ", ".join(f"{label(v)} ({acc[v][0]:.4f} vs {acc[v][1]:.4f})" for v in not_above)
                     + (" view." if len(not_above) == 1 else " views."))
        top1_b = _num(_get(bench, "overall", "honest", "top1_is_b"))
        honest = _num(_get(bench, "overall", "honest", "acc", "mean"))
        if top1_b is not None:
            s.append(f"Without excluding the query words (the honest protocol of Linzen 2016) the accuracy of the "
                     f"{label(best)} view is {_f(honest)} and the top-1 candidate is b itself for {top1_b:.3f} of "
                     "the questions.")
    for bench_name, pretty in (("simlex999", "SimLex-999"), ("wordsim353", "WordSim-353")):
        rho = {v: _get(b, "similarity", bench_name) for v, b in batteries.items()}
        rho = {v: r for v, r in rho.items() if _ok(r) and _num(r.get("rho")) is not None}
        if rho:
            best = max(rho, key=lambda v: rho[v]["rho"])
            worst = min(rho, key=lambda v: rho[v]["rho"])
            r = rho[best]
            s.append(f"On {pretty} the highest Spearman correlation with human ratings, a "
                     f"{_selection_tail(len(rho))}, is ρ = {r['rho']:.3f} [{_f(r.get('ci_lo'))}, "
                     f"{_f(r.get('ci_hi'))}] ({label(best)}, n = {r['n']}), the lowest "
                     f"{rho[worst]['rho']:.3f} ({label(worst)}).")
    pur = {v: _get(b, "fields", "purity") for v, b in batteries.items() if _ok(b.get("fields"))}
    pur = {v: p for v, p in pur.items() if isinstance(p, Mapping) and _num(p.get("purity")) is not None}
    if pur:
        hi = max(pur, key=lambda v: pur[v]["purity"])
        lo = min(pur, key=lambda v: pur[v]["purity"])
        s.append(f"Semantic-field kNN purity@{pur[hi]['k']} over {pur[hi]['n']} lexicon words ranges from "
                 f"{pur[lo]['purity']:.3f} ({label(lo)}) to {pur[hi]['purity']:.3f} ({label(hi)}, a "
                 f"{_selection_tail(len(pur))}; z = {_f(pur[hi].get('z'), 1)} against a label-permutation null "
                 f"of mean {_f(pur[hi].get('null_mean'))}).")
    probe = {v: _get(b, "geometry", "frequency_probe") for v, b in batteries.items()}
    probe = {v: p for v, p in probe.items() if _ok(p) and _num(p.get("r2_mean")) is not None}
    if probe:
        hi = max(probe, key=lambda v: probe[v]["r2_mean"])
        lo = min(probe, key=lambda v: probe[v]["r2_mean"])
        s.append(f"A cross-validated ridge probe recovers log training count with R² between "
                 f"{probe[lo]['r2_mean']:.3f} ({label(lo)}) and {probe[hi]['r2_mean']:.3f} ({label(hi)}); the "
                 f"shuffled-target control of the latter is {_f(probe[hi].get('control_r2_mean'))}.")
    return " ".join(s)


def _sigma_summary(res: SectionResult, run: Any) -> str:
    m = _get(res.metrics, "banks", "prior")
    if not isinstance(m, Mapping):
        return ""
    s: List[str] = []
    disp, init = m.get("dispersion", {}), m.get("init", {})
    q = disp.get("log_var_quantiles", {})
    cv = _num(disp.get("cv_trace"))
    cv_d0 = _num(disp.get("cv_trace_d0"))
    if cv is not None:
        s.append(f"Over the {m.get('n_candidates')} candidates the coefficient of variation of tr Σ is {cv:.3f}"
                 + (f" in the trained coordinates ({cv_d0:.3f} in D0 units)" if cv_d0 is not None else "")
                 + f", which {'passes' if cv > 0.10 else 'fails'} the pre-registered gate CV > 0.10; the per-coordinate "
                 f"log-variances span {_f(q.get('0.01'))} to {_f(q.get('0.99'))} (1st to 99th percentile, median "
                 f"{_f(q.get('0.5'))}) against the initial value {_f(init.get('log_var_init'))}, and a fraction "
                 f"{_f(init.get('frac_at_init_candidates'))} of the candidate rows is still at initialization.")
    freq = _get(m, "frequency", "logdet_vs_log_count")
    if isinstance(freq, Mapping) and _num(freq.get("rho")) is not None:
        s.append(f"log det Σ_v correlates with log training count at Spearman ρ = {freq['rho']:.3f} "
                 f"[{_f(freq.get('ci_lo'))}, {_f(freq.get('ci_hi'))}].")
    ent = m.get("entailment", {})
    if ent.get("status") == "ok":
        kl, ld = ent["kl_asymmetry"], ent["logdet_baseline"]
        fb = ent.get("frequency_baseline", {})
        clusters = ent.get("n_clusters")
        text = (f"On {ent['n']} hyponym–hypernym pairs the KL asymmetry KL(hypo‖hyper) < KL(hyper‖hypo) orders "
                f"{kl['acc']:.3f} [{_f(kl.get('ci_lo'))}, {_f(kl.get('ci_hi'))}] of the pairs correctly"
                + (f" (cluster bootstrap over {clusters} hypernym groups)" if clusters is not None else "")
                + f", against {_f(ld['acc'])} for the larger-log-det baseline")
        text += (f" and {_f(fb['acc'])} for the more-frequent-word baseline." if "acc" in fb else ".")
        s.append(text)
    sim = m.get("similarity", {})
    bench = "simlex999" if _get(sim, "simlex999", "status") == "ok" else next(
        (b for b, v in sim.items() if isinstance(v, Mapping) and v.get("status") == "ok"), None)
    per = _get(sim, bench, "scorers") if bench is not None else None
    head_key = _get(sim, bench, "headline_scorer") if bench is not None else None
    best = _get(sim, bench, "best_gaussian") if bench is not None else None
    key = head_key if isinstance(per, Mapping) and head_key in per else best
    if isinstance(per, Mapping) and key in per:
        delta = per[key].get("delta_vs_euclid_mu", {})
        role = ("the pre-registered Gaussian scorer" if key == head_key else
                f"the best Gaussian scorer ({sim[bench].get('best_gaussian_note', 'post hoc')})")
        text = (f"On {bench} (n = {sim[bench]['n']}) {role}, {per[key]['label']}, reaches "
                f"ρ = {_f(per[key]['rho'])} against {_f(per['cos_mu']['rho'])} for cos(μ) and "
                f"{_f(per['neg_euclid_mu']['rho'])} for the Σ-free control −‖Δμ‖; its paired bootstrap Δρ against "
                f"that control is {_f(delta.get('delta'))} [{_f(delta.get('ci_lo'))}, {_f(delta.get('ci_hi'))}].")
        if key == head_key and best in per and best != key:
            text += (f" The best of the Gaussian scorers is {per[best]['label']} with ρ = {_f(per[best]['rho'])} "
                     f"({sim[bench].get('best_gaussian_note', 'post hoc')}).")
        s.append(text)
    geo = m.get("table_geometry", {})
    if _num(geo.get("effective_rank")) is not None:
        s.append(f"The centered log-variance table has effective rank {geo['effective_rank']:.1f} of K = {run.K} "
                 f"and linear CKA(μ, log σ²) = {_f(geo.get('cka_mu_logvar'))}.")
    rc = res.metrics.get("routing_covariance", {})
    shares = _get(rc, "variance_shares")
    if _get(rc, "status") == "ok" and isinstance(shares, Mapping) and _num(shares.get("sigma")) is not None:
        s.append(f"For the frame-induced routing covariance C_v = U_v⁻¹ Σ_v U_v⁻ᵀ, the variance of log det C_v "
                 f"splits into a Σ share {_f(shares['sigma'])}, a frame share {_f(shares['frame'])} and a "
                 f"cross share {_f(shares['cross'])}.")
    dec = _get(res.metrics, "banks", "decode")
    if isinstance(dec, Mapping) and _num(_get(dec, "dispersion", "cv_trace")) is not None:
        s.append(f"The Gaussian decode bank has CV(tr Σ) = {_f(_get(dec, 'dispersion', 'cv_trace'))} and "
                 f"τ_eff = {_f(dec.get('tau_eff'), 4)}.")
    return " ".join(s)


def _phi_summary(res: SectionResult) -> str:
    m = res.metrics
    if not m:
        return ""
    s: List[str] = []
    chart = m.get("chart", {})
    if chart:
        converged, residual = _barycenter_status(chart)
        s.append(f"The gauge-fixed chart covers {chart.get('n')} candidate frames, a fraction "
                 f"{_f(chart.get('ok_fraction'))} of which have a real principal logarithm; the barycenter over "
                 f"the {chart.get('n_ref')} most frequent frames "
                 + ("converged in every block." if converged is not False else
                    "did not converge in every block (best iterate used"
                    + (f", largest residual {residual:.1e}" if residual is not None else "")
                    + "), so the chart's invariance holds only up to that residual."))
    comm = m.get("commutator_raw", {})
    iid = comm.get("iid_gaussian_ratio") or []
    if _num(comm.get("ratio_block_median")) is not None:
        ref = f"{min(iid):.3f}" if iid and min(iid) == max(iid) else (
            f"{min(iid):.3f}–{max(iid):.3f}" if iid else "–")
        s.append(f"In the raw chart the median per-block commutator ratio ‖[X_a, X_b]‖/(‖X_a‖‖X_b‖) is "
                 f"{comm['ratio_block_median']:.3f}, against √(2/d) = {ref} for independent Gaussian matrices of "
                 f"the same block size, and the median BCH gap ‖log(e^(X_a) e^(−X_b)) − (X_a − X_b)‖/‖X_a − X_b‖ is "
                 f"{_f(_get(comm, 'bch_gap', 'median'))}.")
    ga = _get(m, "group_analogies", "benchmark")
    ov = _get(ga, "overall")
    if isinstance(ov, Mapping) and ov:
        paired = _num(_get(ov, "chart_reranked", "acc"))
        s.append(f"Group analogies Û_(a*) Û_a⁻¹ Û_b retrieve the answer at rank 1 for {_f(_get(ov, 'chart', 'acc'))} "
                 f"of the {ga.get('n_covered')} covered Google questions by chart distance, against "
                 f"{_f(_get(ov, 'only_b', 'acc'))} for ONLY-B; on the {ga.get('n_reranked')} questions of the "
                 f"log-residual rerank the accuracy is {_f(_get(ov, 'rerank', 'acc'))} after the rerank"
                 + (f" and {paired:.3f} by chart distance on the same questions." if paired is not None else "."))
    rel = m.get("relation_transports", {})
    ok_rel = {k: v for k, v in rel.items() if isinstance(v, Mapping) and v.get("status") == "ok"}
    if ok_rel:
        sig = [k for k, v in ok_rel.items() if _num(v.get("p")) is not None and v["p"] < 0.01]
        cons = [v["consistency"] for v in ok_rel.values() if _num(v.get("consistency")) is not None]
        s.append(f"{len(sig)} of {len(ok_rel)} relations have relation transports Û_(a*_i) Û_(a_i)⁻¹ more consistent "
                 f"than their re-pairing null at p < 0.01"
                 + (f", with absolute consistencies between {min(cons):.3f} and {max(cons):.3f}." if cons else "."))
    pos = m.get("positional")
    if isinstance(pos, Mapping) and _num(pos.get("r2_offset")) is not None:

        def vs_null(value: Any, null: Any) -> str:
            if not isinstance(null, Mapping) or _num(null.get("mean")) is None:
                return _f(value)
            return (f"{_f(value)} (position-permutation null mean {_f(null.get('mean'))}, 95th percentile "
                    f"{_f(null.get('q95'))}, p = {_f(null.get('p'))})")

        null = pos.get("null")
        if isinstance(null, Mapping):
            text = (f"For the positional frames the fraction of spectral-feature variance explained by the offset "
                    f"m − n is {vs_null(pos['r2_offset'], null.get('offset'))}, and by the absolute position m "
                    f"{vs_null(pos.get('r2_absolute'), null.get('absolute'))}")
        else:
            text = (f"For the positional frames the fraction of spectral-feature variance explained by the offset "
                    f"m − n is {pos['r2_offset']:.3f} (i.i.d. reference (G − 1)/(N − 1) = {_f(pos.get('r2_chance'))}, "
                    f"which does not apply to the grouping by m), and by the absolute position m "
                    f"{_f(pos.get('r2_absolute'))}")
        outliers = pos.get("outlier_positions")
        if outliers is not None and len(outliers) and _num(pos.get("r2_offset_interior")) is not None:
            interior = pos.get("null_interior") if isinstance(pos.get("null_interior"), Mapping) else {}
            text += (f"; excluding the {len(outliers)} outlying positions these are "
                     f"{vs_null(pos['r2_offset_interior'], interior.get('offset'))} and "
                     f"{vs_null(pos.get('r2_absolute_interior'), interior.get('absolute'))}")
        s.append(text + ".")
    inv = m.get("invariants", {})
    rho = {k: _num(_get(inv, k, "spearman_log_count", "rho")) for k in ("trace", "rotation", "rms_log_sv")}
    if any(v is not None for v in rho.values()):
        s.append(f"The Spearman correlation of the frame invariants with log count is {_f(rho['trace'])} for "
                 f"log det Û, {_f(rho['rotation'])} for the rotation angle and {_f(rho['rms_log_sv'])} for the RMS "
                 "log singular value.")
    return " ".join(s)


def _contextual_summary(res: SectionResult) -> str:
    r"""Prose of :func:`vfe3.semantics.contextual.contextual_summary` (imported lazily); empty if absent."""
    from vfe3.semantics import contextual

    summarize = getattr(contextual, "contextual_summary", None)
    return "" if summarize is None else str(summarize(res.metrics) or "")


# ----------------------------------------------------------------------------------------------
# At a glance
# ----------------------------------------------------------------------------------------------

_GLANCE_GROUPS: Tuple[Tuple[str, str], ...] = (
    ("mu", "μ views"), ("sigma", "Σ views"), ("phi", "φ views"), ("decode", "decode views"),
)


def _glance_group(names: Sequence[str], batteries: Mapping[str, Mapping[str, Any]],
                  label: Callable[[str], str]) -> List[str]:
    r"""Clauses for one channel group: best field purity vs its null, best 3CosAdd − ONLY-B margin,
    best SimLex ρ, each naming the view it comes from (maxima over ``names``)."""
    parts: List[str] = []
    pur = {v: _get(batteries[v], "fields", "purity") for v in names if _ok(batteries[v].get("fields"))}
    pur = {v: p for v, p in pur.items() if isinstance(p, Mapping) and _num(p.get("purity")) is not None}
    if pur:
        v = max(pur, key=lambda n: pur[n]["purity"])
        p = pur[v]
        parts.append(f"field kNN purity@{p.get('k', '?')} up to {p['purity']:.3f} against a label-permutation null "
                     f"of {_f(p.get('null_mean'))} (z = {_f(p.get('z'), 1)}; {label(v)})")
    acc = _analogy_accuracies({v: batteries[v] for v in names})
    margin = {v: a - ob for v, (a, ob) in acc.items() if ob is not None}
    if margin:
        v = max(margin, key=margin.get)
        a, ob = acc[v]
        if margin[v] > 0.0:
            parts.append(f"Google 3CosAdd exceeds ONLY-B by up to {margin[v]:.4f} ({a:.4f} vs {ob:.4f}; {label(v)})")
        else:
            parts.append(f"Google 3CosAdd does not exceed ONLY-B in any of these views (largest margin "
                         f"{margin[v]:+.4f}: {a:.4f} vs {ob:.4f}; {label(v)})")
    rho = {v: _get(batteries[v], "similarity", "simlex999") for v in names}
    rho = {v: r for v, r in rho.items() if _ok(r) and _num(r.get("rho")) is not None}
    if rho:
        v = max(rho, key=lambda n: rho[n]["rho"])
        r = rho[v]
        lo = _num(r.get("ci_lo"))
        parts.append(f"SimLex-999 ρ up to {r['rho']:.3f} [{_f(lo)}, {_f(r.get('ci_hi'))}]"
                     + ("; its 95% CI includes 0" if lo is not None and lo <= 0.0 else "") + f" ({label(v)})")
    return parts


def _glance_contextual(res: Optional[SectionResult]) -> Optional[str]:
    r"""Final-layer distilled-view numbers of the contextual section (retrieval among the collected words)."""
    if res is None:
        return None
    sem = res.metrics.get("layerwise_semantics") if isinstance(res.metrics, Mapping) else None
    if res.status != "ok" or not _ok(sem) or not sem.get("layers"):
        if res.status != "ok":
            why = f"section {res.status}"
        elif isinstance(sem, Mapping) and sem.get("status") in ("failed", "skipped"):
            why = f"layerwise_semantics {sem['status']}: {_last_line(sem.get('reason') or sem.get('error'))}"
        else:
            why = "the distilled-layer battery did not run"
        return f"Contextual: no distilled-layer result ({why}; see Failures and skips)."
    last = sem["layers"][-1]
    parts: List[str] = []
    fields = last.get("fields")
    if _ok(fields) and _num(fields.get("purity")) is not None:
        parts.append(f"field kNN purity {fields['purity']:.3f} against a label-permutation null of "
                     f"{_f(fields.get('null_mean'))} (z = {_f(fields.get('z'), 1)})")
    an = last.get("analogies")
    a = _num(_get(an, "overall", "3cosadd", "mean")) if _ok(an) else None
    ob = _num(_get(an, "overall", "only_b", "mean")) if _ok(an) else None
    if a is not None and ob is not None:
        parts.append(f"Google 3CosAdd {a:.3f} against ONLY-B {ob:.3f} on {an.get('n_covered')} questions"
                     + ("" if a > ob else ", so 3CosAdd does not exceed ONLY-B"))
    static = sem.get("static_mu") or {}
    sa = _num(_get(static, "analogies", "overall", "3cosadd", "mean")) if _ok(static.get("analogies")) else None
    sb = _num(_get(static, "analogies", "overall", "only_b", "mean")) if _ok(static.get("analogies")) else None
    sp = _num(_get(static, "fields", "purity")) if _ok(static.get("fields")) else None
    if sa is not None or sp is not None:
        parts.append("the static prior μ of the same words gives purity " + _f(sp) + " and 3CosAdd "
                     + _f(sa) + " against ONLY-B " + _f(sb))
    if not parts:
        return "Contextual: the distilled-layer battery produced no purity or analogy number."
    return (f"Contextual (final layer {last.get('layer')}, occurrence means distilled over {sem.get('n_words')} "
            "words, retrieval among those words only): " + "; ".join(parts)
            + "; SimLex-999 is not computed for the distilled views.")


def _at_a_glance(results: Mapping[str, SectionResult]) -> str:
    r"""One data-only sentence per channel group (μ, Σ, φ, decode, contextual).

    For the view groups every number is the maximum over the group's views, named in parentheses:
    semantic-field kNN purity against its label-permutation null, the margin of Google 3CosAdd over
    ONLY-B, and the SimLex-999 Spearman ρ with its bootstrap CI. The contextual sentence reports the
    final layer's distilled view.
    """
    lines: List[str] = []
    emb = results.get("embeddings")
    batteries = emb.metrics.get("batteries", {}) if emb is not None and isinstance(emb.metrics, Mapping) else {}
    meta = emb.metrics.get("views", {}) if emb is not None and isinstance(emb.metrics, Mapping) else {}

    def label(v: str) -> str:
        return meta.get(v, {}).get("label", v)

    if emb is not None:
        for group, title in _GLANCE_GROUPS:
            names = [v for v in batteries if meta.get(v, {}).get("group") == group]
            if not names:
                lines.append(f"{title}: none analyzed.")
                continue
            parts = _glance_group(names, batteries, label)
            lines.append(f"{title} ({', '.join(label(v) for v in names)}): "
                         + ("; ".join(parts) if parts else "no purity, analogy or SimLex result") + ".")
    contextual = _glance_contextual(results.get("contextual"))
    if contextual is not None:
        lines.append(contextual)
    if not lines:
        return "_Neither the embedding nor the contextual section ran._\n"
    lead = ("Each view-group number below is the maximum over the views of that group (named in parentheses), "
            "not adjusted for selection.\n\n" if emb is not None else "")
    return lead + "\n".join(f"- {line}" for line in lines) + "\n"


# ----------------------------------------------------------------------------------------------
# Methods, failures
# ----------------------------------------------------------------------------------------------

def _candidates_sentences(run: Any, ctx: AnalysisContext) -> str:
    r"""How the candidate vocabulary was formed: count provenance (``run.counts_source``) and the BPE
    prefix filter as recorded in ``ctx.coverage["candidates"]``."""
    cand = ctx.coverage.get("candidates", {}) if isinstance(ctx.coverage, Mapping) else {}
    english = bool(getattr(ctx.vocab, "english", True))
    base = ("whole-word tokens" if english else
            "valid-text, non-whitespace tokens (the vocabulary is not English, so no whole-word mask is used)")
    source = getattr(run, "counts_source", "unavailable")
    if source == "unigram_log_prior":
        text = (f"Candidates are {base} whose training count, recovered by inverting the checkpoint's add-one "
                "unigram table, reaches the frequency floor; resource words that pass the floor are always kept.")
    elif source == "train_cache_bincount":
        text = (f"Candidates are {base} whose training count, a bincount of the full cached training stream (a "
                "run trained with a token cap saw only a prefix of it), reaches the frequency floor; resource "
                "words that pass the floor are always kept.")
    else:
        text = (f"Training counts are unavailable, so the candidates are all {base} (no frequency floor, id "
                "order, no truncation).")
    pf = cand.get("prefix_filter") if isinstance(cand, Mapping) else None
    if isinstance(pf, Mapping) and pf.get("applied"):
        text += (" Byte-pair tokens that look like whole words but are followed by a continuation piece in more "
                 f"than a fraction {pf.get('max_continuation_fraction')} of their training occurrences (word "
                 f"prefixes such as \" Squ\" + \"irrel\") are removed from the candidates ({pf.get('n_removed', 0)} "
                 "tokens); curated and benchmark words are exempt.")
    return text


def _last_line(text: Any) -> str:
    lines = [line.strip() for line in str(text or "").strip().splitlines() if line.strip()]
    return lines[-1] if lines else "unknown error"


def _failed_steps(obj: Any) -> List[Tuple[str, str]]:
    r"""``(step path, short error)`` of every recorded failure under a section's metrics, depth first.

    Collected: the entries of every ``failed_steps`` list (``{"step", "error"}``), every mapping with
    ``status == "failed"`` and every traceback under ``output_errors`` / ``figure_errors``. Paths are
    ``/``-joined keys; a path already recorded is not repeated.
    """
    found: Dict[str, str] = {}

    def walk(node: Any, path: Tuple[str, ...]) -> None:
        if isinstance(node, Mapping):
            steps = node.get("failed_steps")
            if isinstance(steps, (list, tuple)):
                for entry in steps:
                    if isinstance(entry, Mapping):
                        found.setdefault("/".join((*path, str(entry.get("step", "?")))), _last_line(entry.get("error")))
            for key, value in node.items():
                if key == "failed_steps":
                    continue
                where = (*path, str(key))
                if key in ("output_errors", "figure_errors") and isinstance(value, Mapping):
                    for name, tb in value.items():
                        found.setdefault("/".join((*where, str(name))), _last_line(tb))
                    continue
                if isinstance(value, Mapping) and value.get("status") == "failed":
                    found.setdefault("/".join(where), _last_line(value.get("error") or value.get("reason")))
                if isinstance(value, (Mapping, list, tuple)):
                    walk(value, where)
        elif isinstance(node, (list, tuple)):
            for i, value in enumerate(node):
                if isinstance(value, (Mapping, list, tuple)):
                    walk(value, (*path, str(i)))

    walk(obj, ())
    return list(found.items())


def _failure_lines(results: Mapping[str, SectionResult]) -> List[str]:
    r"""Every failure and skip of the run for REPORT.md: failed/skipped sections, failed views, and the
    failed sub-steps of every section and view battery (:func:`_failed_steps`), plus the skipped
    top-level steps of sections and batteries."""
    lines: List[str] = []
    for name, res in results.items():
        if res.status == "failed":
            lines.append(f"- section **{name}** failed: `{_md_cell(_last_line(res.error))}`")
        elif res.status == "skipped":
            lines.append(f"- section **{name}** skipped: " + _md_cell("; ".join(res.notes) or "no reason recorded"))
        metrics = res.metrics if isinstance(res.metrics, Mapping) else {}
        if name == "embeddings":
            for view, tb in (metrics.get("failed_views") or {}).items():
                lines.append(f"- view **{view}** failed: `{_md_cell(_last_line(tb))}`")
        for path, error in _failed_steps(metrics):
            parts = path.split("/")
            if name == "embeddings" and parts[0] == "batteries" and len(parts) >= 3:
                lines.append(f"- view **{parts[1]}**, step `{'/'.join(parts[2:])}` failed: `{_md_cell(error)}`")
            else:
                lines.append(f"- section **{name}**, step `{path}` failed: `{_md_cell(error)}`")
        tops = ([(f"view {v}, step {k}", value) for v, b in (metrics.get("batteries") or {}).items()
                 if isinstance(b, Mapping) for k, value in b.items()] if name == "embeddings" else
                [(f"section {name}, step {k}", value) for k, value in metrics.items()])
        for where, value in tops:
            if isinstance(value, Mapping) and value.get("status") == "skipped":
                lines.append(f"- {where} skipped: {_md_cell(value.get('reason') or '')}")
    return lines


def _coverage_rows(run: Any, ctx: AnalysisContext) -> List[Tuple[str, str]]:
    cov = ctx.coverage
    rows: List[Tuple[str, str]] = []

    def pair(entry: Any) -> Optional[str]:
        if not isinstance(entry, Mapping) or "n_total" not in entry:
            return None
        if not entry["n_total"]:
            return f"{entry['n_resolved']} / {entry['n_total']}"
        return (f"{entry['n_resolved']} / {entry['n_total']} "
                f"({100.0 * entry['n_resolved'] / entry['n_total']:.1f}%)")

    for key, label in (("fields", "semantic-field words"), ("relations", "relation pairs"),
                       ("curated_analogies", "curated analogies"), ("google_analogies", "Google analogies"),
                       ("hypernyms", "hyponym–hypernym pairs"), ("probe_words", "probe words")):
        text = pair(cov.get(key))
        if text is not None:
            rows.append((label, text))
    for bench, entry in (cov.get("similarity") or {}).items():
        text = pair(entry)
        if text is not None:
            rows.append((bench, text))
    return rows


def _render_report(run: Any, ctx: AnalysisContext, results: Mapping[str, SectionResult],
                   timings: Mapping[str, Any], report_notes: Sequence[str],
                   started: datetime.datetime) -> str:
    cfg = run.cfg
    out = Path(ctx.out_dir)
    lines: List[str] = []
    add = lines.append

    add(f"# Semantic geometry of `{run.run_dir.name}`\n")
    add(f"Run directory `{run.run_dir}`, checkpoint `{run.checkpoint_path.name}`, dataset "
        f"`{run.dataset}` ({run.vocab.kind} tokenizer, V = {run.vocab_size}); analysis started "
        f"{started.isoformat(timespec='seconds')} and took {timings.get('total', 0.0):.0f} s. Sections: "
        + ", ".join(f"{name} **{res.status}**" for name, res in results.items()) + ".\n")

    # ---- model facts --------------------------------------------------------------------------
    cand = ctx.coverage.get("candidates", {})
    counts = run.counts_diagnostics
    facts: List[Tuple[str, Any]] = [
        ("K / layers / heads", f"{run.K} / {run.n_layers} / {run.n_heads} (frame blocks "
                               f"{[s.stop - s.start for s in run.block_slices]})"),
        ("family", f"{cfg.family} (sigma_init {getattr(cfg, 'sigma_init', None)})"),
        ("gauge group", f"{getattr(cfg, 'gauge_group', None)} (n_gen = {run.n_gen}, frames "
                        f"{'active' if run.frames_active else 'inactive'})"),
        ("transport", f"gauge_transport={getattr(cfg, 'gauge_transport', None)}, "
                      f"flat connection, pos_phi={getattr(cfg, 'pos_phi', None)}"),
        ("attention mode", getattr(cfg, "attention_mode", None)),
        ("coordinate normalization", f"block={getattr(cfg, 'norm_type_block', None)}, "
                                     f"affine={getattr(cfg, 'layernorm_affine', None)}"),
        ("decode", f"{cfg.decode_mode} ({run.decode_kind}; unigram prior "
                   f"{getattr(cfg, 'decode_unigram_prior', None)})"),
        ("Σ frozen at encode", run.sigma_frozen),
        ("φ frozen in context", run.phi_frozen),
        ("counts source", f"{run.counts_source}"
                          + (f" (N = {counts['N']}, {counts.get('n_zero')} zero-count tokens)" if "N" in counts else "")),
        ("candidates", f"{int(ctx.candidate_ids.size)} ("
                       + (f"count ≥ {cand.get('min_count')}" if cand.get("min_count") is not None
                          else "no frequency floor")
                       + f", top {cand.get('n_top')} + {cand.get('n_forced')} resource words)"),
    ]
    facts += [(f"coverage: {label}", text) for label, text in _coverage_rows(run, ctx)]
    add("## Model facts\n")
    add(markdown_table([{"fact": k, "value": "unknown" if v is None else v} for k, v in facts],
                       ["fact", "value"]))

    add("## At a glance\n")
    try:
        add(_at_a_glance(results))
    except Exception as exc:                            # prose only; never lose the report
        add(f"(at-a-glance summary could not be generated: {type(exc).__name__}: {exc})\n")

    phi_res = results.get("phi")
    chart = phi_res.metrics.get("chart") if phi_res is not None and isinstance(phi_res.metrics, Mapping) else None
    emb = results.get("embeddings")
    if chart is None and emb is not None and isinstance(emb.metrics, Mapping):
        chart = emb.metrics.get("phi_log_chart")
    add("## What is and is not gauge-meaningful\n")
    add(_gauge_paragraph(run, chart) + "\n")

    # ---- headline tables ----------------------------------------------------------------------
    add("## Headline results\n")
    if emb is not None and emb.status != "failed" and emb.metrics.get("batteries"):
        add("### Embedding views\n")
        add(_headline_tables(emb, ctx))
    for name in ("sigma", "phi", "contextual"):
        res = results.get(name)
        if res is not None and res.headline:
            add(f"### {_SECTION_TITLES[name]}\n")
            add(_kv_table(res.headline))

    # ---- per-section summaries ----------------------------------------------------------------
    add("## Section summaries\n")
    summaries = {
        "embeddings": lambda r: _embeddings_summary(r, ctx),
        "sigma": lambda r: _sigma_summary(r, run),
        "phi": lambda r: _phi_summary(r),
        "contextual": lambda r: _contextual_summary(r),
    }
    for name in SECTIONS:
        res = results.get(name)
        if res is None:
            continue
        add(f"### {_SECTION_TITLES[name]} ({res.status}, {res.seconds or 0.0:.0f} s)\n")
        if res.status == "failed":
            last = (res.error or "").strip().splitlines()
            add(f"This section failed: `{_md_cell(last[-1] if last else 'unknown error')}`. The full traceback "
                f"is in `metrics.json` under `sections.{name}.error`.\n")
        elif res.status == "skipped":
            add("This section was skipped.\n")
        else:
            try:
                text = summaries[name](res)
            except Exception as exc:                    # the summary is prose; never lose the report
                text = f"(summary could not be generated: {type(exc).__name__}: {exc})"
            if text:
                add(text + "\n")
            elif res.headline:
                add("No prose summary is available for this section; the headline table above lists its results "
                    f"and the full numbers are in `metrics.json` under `sections.{name}.metrics`.\n")
        if name == "embeddings" and res.metrics.get("views"):
            view_rows = [{"view": n, "label": v["label"], "metric": v["metric"], "rows": v["n"], "D": v["dim"],
                          "chart status": v["note"]} for n, v in res.metrics["views"].items()]
            add("Views:\n")
            add(markdown_table(view_rows, ["view", "label", "metric", "rows", "D", "chart status"]))
        if res.notes:
            add("Notes:\n")
            for note in dict.fromkeys(res.notes):
                add(f"- {_md_cell(note)}")
            add("")

    # ---- figures and tables -------------------------------------------------------------------
    add("## Figures\n")
    any_figure = False
    for name in SECTIONS:
        res = results.get(name)
        if res is None:
            continue
        pngs = [Path(p) for p in res.figures if Path(p).suffix == ".png"]
        if not pngs:
            continue
        any_figure = True
        add(f"### {_SECTION_TITLES[name]}\n")
        for png in pngs:
            rel = _relative(png, out)
            fig_rel = _relative(png, Path(ctx.figures_dir))
            pdf = png.with_suffix(".pdf")
            pdf_link = f" ([pdf]({_relative(pdf, out)}))" if pdf.is_file() else ""
            add(f"- [{fig_rel}]({rel}){pdf_link}: {_caption(name, fig_rel)}.")
        add("")
    if not any_figure:
        add("_No figures were written._\n")

    add("## Tables\n")
    any_table = False
    for name in SECTIONS:
        res = results.get(name)
        if res is None or not res.tables:
            continue
        entries = []
        for key in res.tables:
            path = Path(ctx.tables_dir) / name / f"{key}.md"
            entries.append(f"[{name}/{key}]({_relative(path, out)})" if path.is_file() else f"{name}/{key}")
        any_table = True
        add(f"- {_SECTION_TITLES[name]}: " + ", ".join(entries))
    add("" if any_table else "_No tables were written._\n")

    # ---- methods ------------------------------------------------------------------------------
    add("## Methods and references\n")
    add(_candidates_sentences(run, ctx) + " Analogies use 3CosAdd (Mikolov, Yih & Zweig, NAACL 2013), 3CosMul and PairDirection (Levy & Goldberg, CoNLL 2014), "
        "and the ONLY-B, IGNORE-A and honest no-exclusion protocols (Linzen, RepEval 2016) on the Google set "
        "(Mikolov et al., arXiv:1301.3781); because each Google category is built from a few dozen word pairs, "
        "analogy intervals resample word pairs within categories (a pair-level bootstrap) and fall back to a "
        "Wilson interval on the number of pairs at zero or full accuracy. Relation offsets are scored by the "
        "offset concentration score against a re-pairing null (Fournier, Dupoux & Dunbar, CoNLL 2020), "
        "leave-one-out projections against a random-word-pair baseline and concept directions under the causal "
        "inner product Cov⁻¹ (Park, Choe & Veitch, ICML 2024), and the "
        "top-PC share (Bolukbasi et al., NeurIPS 2016); semantic projection follows Grand et al. (Nature Human "
        "Behaviour 2022). Similarity uses SimLex-999 (Hill, Reichart & Korhonen, CL 2015) and WordSim-353 "
        "(Finkelstein et al., ACM TOIS 2002) with bootstrap Spearman intervals. Geometry reports anisotropy "
        "(Ethayarajh, EMNLP 2019), IsoScore (Rudman et al., Findings ACL 2022), effective rank (Roy & Vetterli, "
        "EUSIPCO 2007), TwoNN (Facco et al., Sci. Rep. 2017), hubness (Radovanović et al., JMLR 2010) and rogue "
        "dimensions (Timkey & van Schijndel, EMNLP 2021). Gaussian scorers and KL entailment follow Vilnis & "
        "McCallum (ICLR 2015) and Bražinskas, Havrylov & Titov (COLING 2018); the frame barycenter is the "
        "exponential barycenter of Pennec & Arsigny (Matrix Information Geometry, 2013); relation-transport "
        "spectra follow RotatE (Sun et al., ICLR 2019); representation similarity uses linear CKA (Kornblith et "
        "al., ICML 2019); the per-layer decode is the logit lens (nostalgebraist 2020) and contextuality follows "
        "Ethayarajh (EMNLP 2019). UMAP (McInnes et al., arXiv:1802.03426) panels are always paired with PCA, and "
        "every cluster statistic is computed in the original space.\n")

    # ---- failures, skips, notes, timings -------------------------------------------------------
    add("## Failures and skips\n")
    problems = _failure_lines(results)
    add("\n".join(problems) + "\n" if problems else "No section, view or battery step failed or was skipped.\n")

    run_notes = list(dict.fromkeys([*run.notes, *ctx.coverage.get("notes", []), *report_notes]))
    if run_notes:
        add("## Run notes\n")
        for note in run_notes:
            add(f"- {_md_cell(note)}")
        add("")

    add("## Timings\n")
    rows = [{"stage": "load run", "seconds": float(timings.get("load", 0.0))},
            {"stage": "prepare context", "seconds": float(timings.get("context", 0.0))}]
    rows += [{"stage": f"section {k}", "seconds": float(v or 0.0)} for k, v in timings.get("sections", {}).items()]
    rows.append({"stage": "total", "seconds": float(timings.get("total", 0.0))})
    add(markdown_table(rows, ["stage", "seconds"], digits=1))
    opts = ctx.options
    add(f"Options: preset {opts.get('preset')}, min_count {opts.get('min_count')}, max_candidates "
        f"{opts.get('max_candidates')}, seed "
        f"{ctx.seed}, threads {ctx.threads}, device {ctx.device}, umap {opts.get('umap')}, reference_position "
        f"{opts.get('reference_position')}, frame_ref_count {opts.get('frame_ref_count')}, "
        f"geometry_max_points {opts.get('geometry_max_points')}, headline_view {opts.get('headline_view')}. "
        "Everything else is in `metrics.json`.\n")
    return "\n".join(lines)
