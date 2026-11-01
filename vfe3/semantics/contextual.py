r"""Contextual (per-layer) beliefs of a trained vfe3 model.

The static-table sections read the token prior ``(mu_v, Sigma_v, phi_v)``; this section runs the
model and reads what the stack does with it. A forward pass with ``capture={"diagnostic": {}}``
exposes the stack input :math:`q^{(0)}` (the encoded prior for ``encode_mode="per_token"``) and the
output :math:`q^{(l)} = (\mu^{(l)}, \Sigma^{(l)}, \phi^{(l)})` of every block ``l = 1..L``. On
that per-layer bank the section computes, in the protocols of the transformer-interpretability
literature adapted to Gaussian beliefs:

* the logit lens (nostalgebraist 2020): every layer's belief read out through the model's OWN final
  readout, ``prior_bank.decode`` with the realized frame of the same forward whenever the decoder
  reads it (``VFEModel._decoder_reads_frame``: ``decode_coordinates='reference'``, or a KL decoder of
  ``gaussian_frame_diagonal`` in either coordinate setting) and the unigram bias, exactly the path
  ``VFEModel.forward_beliefs(return_logits=True)`` applies after the last block;
* per-layer geometry (anisotropy, IsoScore, effective rank, TwoNN, rogue dimensions);
* layer-by-layer linear CKA (Kornblith et al. 2019; debiased HSIC of Song et al. 2012), CKA with the
  token prior mean and with the next token's decode row;
* contextuality (Ethayarajh 2019): self-similarity, intra-window similarity and maximum explainable
  variance, each against the layer's random baseline; the Gaussian displacement
  :math:`\mathrm{KL}(q^{(l)}_i \| q^{(0)}_i)` from the prior and the within-type variance share;
* layerwise static distillation (Bommasani, Davis & Cardie 2020): the mean of a word's contextual
  beliefs over sampled training contexts, run through the analogy and field-clustering battery;
* token clouds of polysemous words (Coenen, Reif et al. 2019).

Coordinates. Every cross-row statistic (geometry, CKA, the cosine terms of contextuality, the
occurrence vectors of the distillation and the token clouds) and the within-type variance shares use
the chart the decoder compares with its decode rows: the reference means
:math:`\xi^{(l)}_i = U_i^{-1}\mu^{(l)}_i` of the realized frame under ``decode_coordinates='reference'``
(where the local means depend on the trivialization), the local means otherwise. The KL to the stack
input is chart-free and is evaluated in the family's own geometry: on :math:`(\xi, c)` for
``gaussian_frame_diagonal`` (:math:`\Sigma_i = U_i\operatorname{diag}(c_i)U_i^\top`), with the dense
covariances for a full layout, from the stored diagonals otherwise.

All forward passes run under ``torch.no_grad()`` (``inference_mode`` is rejected by the
conditional-filtering route) in float32 with autocast disabled (``SemanticRun.build_model``).
Figures go to ``figures/contextual/``, tables to ``tables/contextual/``.

Device. The model and every forward run on ``ctx.device``. Each captured forward moves its results
to the host in one transfer per batch (all layers stacked), the logit-lens statistics of all layers
of a batch are reduced on the device and transferred once, and the equality flags that detect a
frozen Σ or φ are accumulated on the device and read once at the end, so no loop over layers
synchronizes with the host. The batch sizes and the lens chunk are device-dependent when their
options are ``None`` (:func:`_auto_batch_windows`, :func:`_auto_lens_chunk_tokens`): the CPU keeps 8
windows per forward and 1,024 lens tokens; a CUDA device takes about 4,096 tokens per captured
forward and sizes the lens chunk to a quarter of the free device memory. A CUDA out-of-memory error
in a forward halves the batch (recorded as ``oom_resplits``), as the exact-symmetry failure does.
"""

from __future__ import annotations

import math
import time
import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from vfe3.semantics import stats, vectors
from vfe3.semantics.common import (
    GAUSSIAN_BANK_DECODE_KINDS,
    OKABE_ITO,
    AnalysisContext,
    SectionResult,
    VectorView,
    apply_style,
    diagonal_redundancy_status,
    markdown_table,
    panel_label,
    save_figure,
    write_table,
)


_SECTION = "contextual"

# Budgets and switches read from ``ctx.options`` (any key may be overridden there). ``None`` batch
# sizes and lens chunk are resolved per device by ``_resolved_options`` (CPU: 8 windows, 1024 tokens).
CONTEXTUAL_DEFAULTS: Dict[str, Any] = {
    "contextual_tokens":     32768,    # bank size (tokens of contextual_split)
    "contextual_split":      "validation",
    "contextual_seq_len":    None,     # None -> cfg.max_seq_len
    "contextual_batch_size": None,     # windows per bank forward (None -> _auto_batch_windows)
    "lens_tokens":           8192,     # tokens decoded at every layer by the logit lens
    "lens_chunk_tokens":     None,     # tokens per lens forward, bounds the (B, N, V) logits
                                       # (None -> _auto_lens_chunk_tokens)
    "contextual_points":     4096,     # token sample for per-layer geometry and CKA
    "min_type_occurrences":  5,        # Ethayarajh: types with at least this many bank occurrences
    "max_type_occurrences":  50,       # occurrences per type used (seeded subsample)
    "occurrence_tokens":     131072,   # forward budget of the occurrence contexts
    "occurrence_split":      "train",
    "occurrence_per_word":   4,
    "occurrence_batch_size": None,     # windows per occurrence forward (None -> _auto_batch_windows)
    "cloud_per_word":        32,       # occurrences per ambiguous word (token clouds)
    "context_len":           48,
    "n_boot":                1000,
}

# Device-dependent forward budgets (used when the options above are None).
_CPU_BATCH_WINDOWS = 8                 # windows per captured forward on the CPU (bank, occurrences)
_CPU_LENS_CHUNK_TOKENS = 1024          # tokens per logit-lens forward on the CPU
_CUDA_FORWARD_TOKENS = 4096            # tokens per captured forward on a CUDA device
_CUDA_LENS_MEMORY_FRACTION = 0.25      # share of the free CUDA memory the lens's (B, N, V) tensors may take
_CUDA_LENS_MAX_CHUNK_TOKENS = 16384    # upper bound of the automatic CUDA lens chunk
_LENS_LIVE_VOCAB_TENSORS = 8           # float32 (B, N, V) tensors alive at once in one lens chunk

# Stable per-step random streams, so each stochastic step draws the same numbers whatever else ran.
_STEP_STREAM = {"bank": 11, "lens": 12, "sample": 13, "geometry": 14, "cka": 15, "contextuality": 16,
                "occurrences": 17, "semantics": 18, "clouds": 19, "sigma": 20}

_CLOUD_MIN_OCCURRENCES = 8
_POSITION_BINS = 16


# ----------------------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------------------

def _opt(ctx: AnalysisContext, key: str) -> Any:
    return ctx.options.get(key, CONTEXTUAL_DEFAULTS[key])


def _rng(ctx: AnalysisContext, step: str) -> np.random.Generator:
    return np.random.default_rng([int(ctx.seed), _STEP_STREAM[step]])


def _attempt(store: Dict[str, Any], key: str, fn: Callable[[], Any], notes: List[str]) -> Any:
    r"""Run one analysis; on failure store a ``failed`` record (error + traceback) and note it."""
    try:
        store[key] = fn()
    except Exception as exc:  # recorded, not swallowed
        store[key] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                      "traceback": traceback.format_exc()[-3000:]}
        notes.append(f"contextual {key} failed: {type(exc).__name__}: {exc}")
    return store[key]


def _ok(result: Any) -> bool:
    return isinstance(result, dict) and result.get("status") not in ("failed", "skipped")


def _failed_steps(obj: Any, path: str = "") -> List[Dict[str, str]]:
    r"""Every ``{"status": "failed", "error": ...}`` record under ``obj`` as ``{"step": path, "error": ...}``.

    ``path`` is dotted with list indices, e.g. ``geometry.layers[3].twonn``; a failed record is not
    descended into.
    """
    if isinstance(obj, Mapping):
        if obj.get("status") == "failed":
            return [{"step": path or _SECTION, "error": str(obj.get("error", "unknown error"))}]
        found: List[Dict[str, str]] = []
        for key, value in obj.items():
            found += _failed_steps(value, f"{path}.{key}" if path else str(key))
        return found
    if isinstance(obj, (list, tuple)):
        return [rec for i, value in enumerate(obj) for rec in _failed_steps(value, f"{path}[{i}]")]
    return []


def _as_numpy_stream(stream: Any) -> np.ndarray:
    r"""The 1-D token stream as a numpy view (no copy of a memory-mapped cache)."""
    tokens = stream.numpy() if isinstance(stream, torch.Tensor) else np.asarray(stream)
    if tokens.ndim != 1:
        raise ValueError(f"token stream must be 1-D, got shape {tokens.shape}")
    return tokens


def _covariance_diagonal(run: Any, sigma: torch.Tensor) -> torch.Tensor:
    r"""``diag Cov`` of a belief's sigma slot, family-aware (a full family stores a matrix).

    For ``gaussian_frame_diagonal`` this is the stored dispersion, the reference variances :math:`c_i`
    (the diagonal of :math:`U_i^{-1}\Sigma_iU_i^{-\top}`, not of the local :math:`\Sigma_i`)."""
    from vfe3.families.base import get_family

    return get_family(run.cfg.family).covariance_diagonal(sigma, eps=run.cfg.eps)


def _stores_reference_variances(run: Any) -> bool:
    r"""Whether the family stores the reference variances :math:`c_i` (``gaussian_frame_diagonal``)."""
    from vfe3.families.base import get_family

    return bool(get_family(run.cfg.family).frame_diagonal_covariance)


def _section_chart(run: Any) -> str:
    r"""``"reference"`` when the decoder compares :math:`\xi_i = U_i^{-1}\mu_i` with its decode rows
    (``decode_coordinates='reference'``, linear or Gaussian-KL), else ``"local"``."""
    return "reference" if getattr(run.cfg, "decode_coordinates", "local") == "reference" else "local"


def _realized_frame(out: Mapping[str, Any]) -> Any:
    r"""The realized relational frame :math:`U_i` of one captured forward: the decode frame when the decoder
    reads it, else the relational frame a frame-diagonal family captures (one object when both exist)."""
    frame = out.get("canonical_frame")
    if frame is None:
        frame = out.get("relational_frame")
    if frame is None:
        raise RuntimeError("the captured forward holds no realized relational frame (neither the decode frame "
                           "nor the frame-diagonal family's relational frame)")
    return frame


def _reference_means(frame: Any, mu: torch.Tensor) -> torch.Tensor:
    r""":math:`\xi_i = U_i^{-1}\mu_i` of (B, N, K) local means through the realized frame, in ``mu``'s dtype."""
    from vfe3.model.frame_actions import frame_matvec

    return frame_matvec(frame.inverse, mu, frame.block_diagonal)


def _chart_covariance_diagonal(belief: Any, frame: Any, *, reference_variances: bool) -> torch.Tensor:
    r"""(B, N, K) covariance diagonal of one belief in the section's chart where it is not the stored dispersion.

    ``gaussian_frame_diagonal`` in the local chart: :math:`\operatorname{diag}(U_i\operatorname{diag}(c_i)U_i^\top)`,
    :math:`(\Sigma_i)_{kk} = \sum_m (U_i)_{km}^2 c_{i,m}`; any other family in the reference chart:
    :math:`\operatorname{diag}(U_i^{-1}\Sigma_iU_i^{-\top})` of the diagonal or full local :math:`\Sigma_i`.
    """
    from vfe3.model.frame_actions import frame_diagonal_local_covariance, reference_variances as reference_diag

    if reference_variances:
        return frame_diagonal_local_covariance(belief.sigma, frame, diagonal=True)
    return reference_diag(belief.mu, belief.sigma, frame)


def _exact_layer_kl(layers: Sequence[Any], frame: Any, *, reference_variances: bool,
                    chunk: int = 4096) -> torch.Tensor:
    r"""(L+1, B N) float64 :math:`\mathrm{KL}(q^{(l)}_i \| q^{(0)}_i)` where the stored diagonals do not determine it.

    ``gaussian_frame_diagonal``: the realized frame is fixed through the stack, so
    :math:`\Sigma^{(l)}_i = U_i\operatorname{diag}(c^{(l)}_i)U_i^\top` and, the KL being invariant under the common
    affine map :math:`U_i^{-1}`,

    .. math:: \mathrm{KL} = \tfrac12\sum_k\Big[\frac{c^{(l)}_k}{c^{(0)}_k} + \frac{(\Delta\xi_k)^2}{c^{(0)}_k} - 1
              - \log\frac{c^{(l)}_k}{c^{(0)}_k}\Big], \qquad \xi = U^{-1}\mu .

    A full covariance (``(B, N, K, K)``) gives the dense Gaussian KL from Cholesky factors,

    .. math:: \tfrac12\big[\operatorname{tr}(\Sigma_0^{-1}\Sigma_l) + \Delta\mu^\top\Sigma_0^{-1}\Delta\mu - K
              + \log\det\Sigma_0 - \log\det\Sigma_l\big].

    A layer whose covariance equals the stack input's bitwise contributes the Mahalanobis term
    :math:`\tfrac12\Delta\mu^\top\Sigma_0^{-1}\Delta\mu` alone. Positions whose covariance has no Cholesky factor are
    NaN. Computed on the device in row chunks of ``chunk`` positions.
    """
    from vfe3.model.frame_actions import frame_matvec

    base = layers[0]
    K = int(base.mu.shape[-1])
    same = [b.sigma is base.sigma or torch.equal(b.sigma, base.sigma) for b in layers]
    if reference_variances:
        inverse = frame.inverse.double()                                           # xi in float64
        xi0 = frame_matvec(inverse, base.mu.double(), frame.block_diagonal).reshape(-1, K)   # (B N, K)
        c0 = base.sigma.reshape(-1, K).double()
        rows: List[torch.Tensor] = []
        for b, frozen in zip(layers, same):
            xi = frame_matvec(inverse, b.mu.double(), frame.block_diagonal).reshape(-1, K)
            kl = 0.5 * ((xi - xi0).square() / c0).sum(-1)
            if not frozen:
                ratio = b.sigma.reshape(-1, K).double() / c0
                kl = kl + 0.5 * (ratio - 1.0 - torch.log(ratio)).sum(-1)
            rows.append(kl)
        return torch.stack(rows)                                                   # (L+1, B N)
    mu0 = base.mu.reshape(-1, K).double()
    S0 = base.sigma.reshape(-1, K, K).double()
    n = int(mu0.shape[0])
    out = torch.empty(len(layers), n, dtype=torch.float64, device=mu0.device)
    for s in range(0, n, chunk):
        sl = slice(s, s + chunk)
        L0, info0 = torch.linalg.cholesky_ex(S0[sl])                               # (c, K, K)
        logdet0 = 2.0 * torch.log(torch.diagonal(L0, dim1=-2, dim2=-1)).sum(-1)
        for l, (b, frozen) in enumerate(zip(layers, same)):
            d = (b.mu.reshape(-1, K)[sl].double() - mu0[sl]).unsqueeze(-1)         # (c, K, 1)
            kl = 0.5 * torch.linalg.solve_triangular(L0, d, upper=False).square().sum((-2, -1))
            if not frozen:
                Ll, info_l = torch.linalg.cholesky_ex(b.sigma.reshape(-1, K, K)[sl].double())
                trace = torch.linalg.solve_triangular(L0, Ll, upper=False).square().sum((-2, -1))
                logdet_l = 2.0 * torch.log(torch.diagonal(Ll, dim1=-2, dim2=-1)).sum(-1)
                kl = kl + 0.5 * (trace - K + logdet0 - logdet_l)
                kl = torch.where(info_l != 0, torch.full_like(kl, float("nan")), kl)
            out[l, sl] = torch.where(info0 != 0, torch.full_like(kl, float("nan")), kl)
    return out


# Relative tolerance of the stack-input covariance diagonal against the token table (float32 model against the
# float64 table accessors), and of the stack-input means (an exact gather for encode_mode='per_token').
_TOKEN_PRIOR_RTOL_VAR = 1e-4
_TOKEN_PRIOR_RTOL_MU = 1e-6


def _token_prior_gap(run: Any, token_ids: np.ndarray, mu0: np.ndarray, var0: np.ndarray) -> Dict[str, Any]:
    r"""How far the stack input (layer 0) lies from the token-table prior of each position's token.

    Compares :math:`\mu^{(0)}_i` with :math:`\mu_{v_i}` (``run.mu``) and the stored covariance diagonal with
    ``exp(run.log_var)`` (the reference variances :math:`c_v` under ``gaussian_frame_diagonal``, as stored). The stack
    input equals the token prior when the largest mean gap is within
    :data:`_TOKEN_PRIOR_RTOL_MU` :math:`\cdot\max(1, \max|\mu_v|)` and the largest relative variance gap within
    :data:`_TOKEN_PRIOR_RTOL_VAR`; ``None`` entries mean the table could not be read.
    """
    out: Dict[str, Any] = {"stack_input_equals_token_prior": None,
                           "stack_input_minus_token_prior_max_abs_mu": None,
                           "stack_input_minus_token_prior_max_rel_var": None}
    if token_ids.size == 0:
        return out
    uniq, inverse = np.unique(token_ids, return_inverse=True)
    try:
        table_mu = np.asarray(run.mu(uniq), dtype=np.float32)[inverse]                # (n, K)
    except (AttributeError, KeyError, ValueError, IndexError):
        return out
    mu_gap = float(np.abs(mu0.astype(np.float64) - table_mu.astype(np.float64)).max())
    var_gap: Optional[float] = None
    try:
        table_var = np.exp(np.asarray(run.log_var(uniq), dtype=np.float64))[inverse]   # (n, K)
        var_gap = float(np.max(np.abs(var0.astype(np.float64) - table_var) / table_var))
    except (AttributeError, KeyError, ValueError, IndexError):
        var_gap = None
    scale = max(1.0, float(np.abs(table_mu).max()))
    out.update({
        "stack_input_equals_token_prior": bool(mu_gap <= _TOKEN_PRIOR_RTOL_MU * scale
                                               and (var_gap is None or var_gap <= _TOKEN_PRIOR_RTOL_VAR)),
        "stack_input_minus_token_prior_max_abs_mu": mu_gap,
        "stack_input_minus_token_prior_max_rel_var": var_gap})
    return out


def _cluster_mean_ci(values: np.ndarray, clusters: np.ndarray, rng: np.random.Generator,
                     n_boot: int = 1000) -> Dict[str, Any]:
    r"""Mean with a 95% percentile CI from resampling whole clusters (windows) with replacement.

    Tokens of one window share their context, so a token-level bootstrap would understate the
    uncertainty; the cluster bootstrap resamples windows and recomputes the pooled mean
    :math:`\sum_{g \in G^*} S_g / \sum_{g \in G^*} n_g`.
    """
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    c = np.asarray(clusters).reshape(-1)
    if v.size == 0:
        raise ValueError("no values")
    if not np.isfinite(v).all():
        raise ValueError("non-finite values")
    _, inverse = np.unique(c, return_inverse=True)
    sums = np.bincount(inverse, weights=v)
    counts = np.bincount(inverse).astype(np.float64)
    G = sums.size
    boot = np.empty(n_boot, dtype=np.float64)
    for start in range(0, n_boot, 200):
        stop = min(n_boot, start + 200)
        idx = rng.integers(0, G, size=(stop - start, G))
        boot[start:stop] = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean": float(v.mean()), "ci_lo": float(lo), "ci_hi": float(hi), "n": int(v.size),
            "n_clusters": int(G)}


def _quantiles(values: np.ndarray) -> Dict[str, float]:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    q = np.percentile(v, [25.0, 50.0, 75.0])
    return {"q25": float(q[0]), "median": float(q[1]), "q75": float(q[2])}


def _layer_names(n_layers: int, encode_mode: str) -> List[str]:
    first = "stack input (encoded prior)" if encode_mode == "per_token" else f"stack input ({encode_mode})"
    return [first] + [f"block {l}" for l in range(1, n_layers + 1)]


def _token_sample(bank: "ContextualBank", ctx: AnalysisContext, max_points: int) -> np.ndarray:
    r"""Sorted seeded sample of bank rows, shared by the geometry and CKA analyses (row-matched)."""
    n = bank.n
    if n <= max_points:
        return np.arange(n, dtype=np.int64)
    return np.sort(_rng(ctx, "sample").choice(n, size=int(max_points), replace=False)).astype(np.int64)


def _auto_batch_windows(device: torch.device, seq_len: int) -> int:
    r"""Windows per captured forward: :data:`_CPU_BATCH_WINDOWS` on the CPU, about
    :data:`_CUDA_FORWARD_TOKENS` tokens (at least one window) on a CUDA device."""
    if torch.device(device).type == "cuda":
        return max(1, _CUDA_FORWARD_TOKENS // max(1, int(seq_len)))
    return _CPU_BATCH_WINDOWS


def _auto_lens_chunk_tokens(device: torch.device, vocab_size: int, seq_len: int,
                            free_bytes: Optional[int] = None) -> int:
    r"""Tokens per logit-lens forward.

    CPU: :data:`_CPU_LENS_CHUNK_TOKENS`. CUDA: the lens holds about
    :data:`_LENS_LIVE_VOCAB_TENSORS` float32 ``(B, N, V)`` tensors at once, so with ``F`` free bytes
    (``torch.cuda.mem_get_info`` unless ``free_bytes`` is given) the chunk is

    .. math:: T = \Big\lfloor \frac{f F}{4 V n_\mathrm{live}} \Big\rfloor,
              \qquad f = 0.25,

    clipped to ``[seq_len, 16384]`` (at least one window).
    """
    device = torch.device(device)
    if device.type != "cuda":
        return _CPU_LENS_CHUNK_TOKENS
    if free_bytes is None:
        free_bytes, _ = torch.cuda.mem_get_info(device)
    per_token = 4 * int(vocab_size) * _LENS_LIVE_VOCAB_TENSORS
    tokens = int(_CUDA_LENS_MEMORY_FRACTION * float(free_bytes)) // per_token
    return int(min(_CUDA_LENS_MAX_CHUNK_TOKENS, max(int(seq_len), tokens)))


def _resolved_options(ctx: AnalysisContext, run: Any) -> Dict[str, Any]:
    r"""Every :data:`CONTEXTUAL_DEFAULTS` key read from ``ctx.options``, with the ``None`` batch sizes
    and lens chunk resolved for ``ctx.device`` (these are the values the section uses and records)."""
    opts = {k: _opt(ctx, k) for k in CONTEXTUAL_DEFAULTS}
    device = torch.device(ctx.device)
    N = int(opts["contextual_seq_len"] if opts["contextual_seq_len"] is not None else run.cfg.max_seq_len)
    if opts["contextual_batch_size"] is None:
        opts["contextual_batch_size"] = _auto_batch_windows(device, N)
    if opts["occurrence_batch_size"] is None:
        opts["occurrence_batch_size"] = _auto_batch_windows(device, int(opts["context_len"]))
    if opts["lens_chunk_tokens"] is None:
        opts["lens_chunk_tokens"] = _auto_lens_chunk_tokens(device, int(run.vocab_size), N)
    return opts


def _equal_flag(a: Optional[torch.Tensor], b: Optional[torch.Tensor], device: torch.device) -> torch.Tensor:
    r"""``torch.equal(a, b)`` as a 0-dim bool tensor on ``device``, without a host synchronization."""
    if a is None or b is None:
        return torch.tensor(a is None and b is None, device=device)
    if a.shape != b.shape:
        return torch.zeros((), dtype=torch.bool, device=device)
    return torch.eq(a, b).all()


def _rowdot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    r""":math:`\sum_v a_{\dots v} b_{\dots v}` over the last dim as a batched matmul, so no ``a * b``
    temporary of the full ``(B, N, V)`` size is allocated. Returns shape ``a.shape[:-1]``."""
    return torch.matmul(a.unsqueeze(-2), b.unsqueeze(-1))[..., 0, 0]


def _snippet(vocab: Any, context: np.ndarray, n_tokens: int = 8) -> str:
    r"""The last ``n_tokens`` of a context as text, the final (occurrence) token bracketed."""
    ids = [int(t) for t in context[-n_tokens:]]
    sep = "" if vocab.is_bpe else " "                                    # BPE pieces carry their spaces
    left = sep.join(vocab.strings[t] for t in ids[:-1])
    last = vocab.strings[ids[-1]]
    lead = (" " if last[:1].isspace() else "") if vocab.is_bpe else " "
    text = f"{left}{lead}[{last.strip()}]"
    return text.replace("\n", "\\n").replace("\t", "\\t").strip()


# ----------------------------------------------------------------------------------------------
# Forward extraction and the model's own readout
# ----------------------------------------------------------------------------------------------

def _forward_layers(model: Any, x: torch.Tensor, *, return_logits: bool = False) -> Dict[str, Any]:
    r"""One captured forward: the stack input and every block output (and optionally the logits).

    ``layers[0]`` is ``capture["diagnostic"]["initial_belief"]`` -- the belief ``vfe_stack``
    receives (after the positional right factor) -- and ``layers[l]`` is ``layer_outputs[l-1]``, the post-block belief
    handed to block ``l+1``; ``layers[L]`` is the raw stack output. ``canonical_frame`` is the decode frame (captured
    whenever the decoder reads it) and ``relational_frame`` the realized relational frame a frame-diagonal family
    captures; both are ``None`` where the model does not capture them.
    """
    capture: Dict[str, Any] = {"diagnostic": {}}
    with torch.no_grad():
        belief, logits = model.forward_beliefs(x, return_logits=return_logits, capture=capture)
    diagnostic = capture["diagnostic"]
    return {
        "layers":           [diagnostic["initial_belief"], *diagnostic["layer_outputs"]],
        "canonical_frame":  capture.get("canonical_frame"),
        "relational_frame": capture.get("relational_frame"),
        "belief":           belief,
        "logits":           logits,
    }


# The zero-jitter Cholesky of the relational score checks the projected covariance for exact
# symmetry (``rtol = 32 eps``, ``atol = 0``, vfe3/relational_scoring.py). On CPU the BLAS summation
# order depends on the batch shape, so a large batch can fail that check although every one of its
# windows passes alone; the per-window results do not depend on the batch composition.
_SYMMETRY_FAILURE = "must be symmetric for zero-jitter Cholesky"


@dataclass
class _SplitLog:
    r"""Re-split bookkeeping of one forward step (bank, logit lens or occurrences)."""

    resplit_batches: int = 0                                          # symmetry-failed multi-window forwards, halved
    oom_resplits:    int = 0                                          # CUDA out-of-memory forwards, halved
    skipped: List[int] = field(default_factory=list)                  # window indices that failed alone

    def record(self) -> Dict[str, Any]:
        return {"resplit_batches": int(self.resplit_batches), "oom_resplits": int(self.oom_resplits),
                "skipped_windows": len(self.skipped), "skipped_window_index": [int(i) for i in self.skipped]}


def _forward_split(model: Any, x: torch.Tensor, log: _SplitLog, *, return_logits: bool = False,
                   offset: int = 0) -> List[Tuple[int, int, Dict[str, Any]]]:
    r"""Captured forwards of the windows ``x`` (B, N), halving the batch on the exact-symmetry
    failure or on a CUDA out-of-memory error.

    Runs :func:`_forward_layers` on the whole batch; when the model raises the ``ValueError`` of its
    zero-jitter Cholesky symmetry check (message containing :data:`_SYMMETRY_FAILURE`), the batch is
    split into halves that are retried recursively, down to single windows. A single window that
    still fails is skipped and its index ``offset + i`` recorded in ``log``. A
    ``torch.cuda.OutOfMemoryError`` of a multi-window batch empties the CUDA cache and halves the
    batch the same way (counted in ``log.oom_resplits``); on a single window it propagates, as does
    any other exception. Returns the successful pieces as ``(start, stop, out)`` with
    ``x[start:stop]`` the windows of ``out``, in ascending order.
    """
    B = int(x.shape[0])
    out_of_memory = False
    try:
        return [(0, B, _forward_layers(model, x, return_logits=return_logits))]
    except ValueError as exc:
        if _SYMMETRY_FAILURE not in str(exc):
            raise
    except torch.cuda.OutOfMemoryError:
        if B == 1:
            raise
        out_of_memory = True
    if out_of_memory:                                  # the handler has released the failed forward's frames
        log.oom_resplits += 1
        if x.device.type == "cuda":
            torch.cuda.empty_cache()
    elif B == 1:
        log.skipped.append(int(offset))
        return []
    else:
        log.resplit_batches += 1
    h = B // 2
    pieces = _forward_split(model, x[:h], log, return_logits=return_logits, offset=offset)
    right = _forward_split(model, x[h:], log, return_logits=return_logits, offset=offset + h)
    return pieces + [(a + h, b + h, out) for a, b, out in right]


def _readout(model: Any, mu: torch.Tensor, sigma: torch.Tensor, canonical_frame: Any,
             *, encoded_sigma: Optional[torch.Tensor] = None) -> torch.Tensor:
    r"""Logits of one belief through the model's final readout, as ``forward_beliefs`` applies it.

    .. math:: \ell = \mathrm{decode}\big(\mu,\ \Sigma;\ U\big),

    i.e. ``VFEModel._decode_belief_with_context``: ``prior_bank.decode`` in the fp32 island with the
    realized frame ``U`` of the same forward whenever ``VFEModel._decoder_reads_frame()`` holds
    (``decode_coordinates='reference'``, or a KL decoder of ``gaussian_frame_diagonal``; ``canonical_frame`` is
    ``None`` otherwise) and the unigram bias :math:`\kappa \log\pi_v`. Under ``decode_covariance='encoded'``
    the decoder reads ``encoded_sigma``, the stack-input covariance (``layers[0].sigma``, the tensor
    ``forward_beliefs`` decodes with; audit 2026-10-16 F64). Returns ``(B, N, V)`` float32 logits.
    """
    with torch.no_grad():
        return model._decode_belief_with_context(mu, sigma, canonical_frame=canonical_frame,
                                                 encoded_sigma=encoded_sigma)


# ----------------------------------------------------------------------------------------------
# The per-layer bank
# ----------------------------------------------------------------------------------------------

@dataclass
class ContextualBank:
    r"""Per-layer beliefs of every position of ``n_seq`` windows of one split, window-major.

    Row ``r`` is position ``positions[r]`` of window ``seq_index[r]`` (``windows[seq_index[r]]``);
    ``targets[r]`` is the next token of the stream (``-1`` if the stream ends). Layer 0 is the stack
    input :math:`q^{(0)}` (the encoded token prior for ``encode_mode="per_token"``), layer ``l >= 1``
    the output of block ``l``. ``sigma`` holds the per-layer covariance diagonal (the stored dispersion:
    the reference variances :math:`c_i` under ``gaussian_frame_diagonal``) and is ``None`` when
    every layer's Sigma is bitwise equal to the stack input's (``sigma_frozen``, measured with
    ``torch.equal``); ``sigma0`` is the stack input's covariance diagonal either way.

    ``chart`` names the coordinates of the cross-row statistics (:func:`_section_chart`): under
    ``"reference"`` ``xi`` holds :math:`\xi^{(l)}_i = U_i^{-1}\mu^{(l)}_i` of the realized frame and
    ``prior_xi`` the token-table means in the same chart, :math:`U_i^{-1}\mu_{v_i}`; both are ``None`` under
    ``"local"``. ``chart_sigma``/``chart_sigma0`` hold the covariance diagonal in that chart where it is not
    the stored dispersion (:func:`_chart_covariance_diagonal`; ``chart_sigma`` is ``None`` when Sigma is
    frozen). ``kl`` is the exact :math:`\mathrm{KL}(q^{(l)}_i\|q^{(0)}_i)` (:func:`_exact_layer_kl`) where the
    stored diagonals do not determine it (the frame-diagonal family, a full covariance), else ``None``.
    """

    token_ids:    np.ndarray                 # (n,) int64
    targets:      np.ndarray                 # (n,) int64 next-token ids, -1 if none
    positions:    np.ndarray                 # (n,) int64
    seq_index:    np.ndarray                 # (n,) int64
    mu:           np.ndarray                 # (L+1, n, K) float32
    sigma:        Optional[np.ndarray]       # (L+1, n, K) float32 covariance diagonal, None if frozen
    phi_moves:    bool
    sigma0:       np.ndarray                 # (n, K) float32 covariance diagonal of the stack input
    windows:      np.ndarray                 # (n_seq, N) int64 input windows
    offsets:      np.ndarray                 # (n_seq,) int64 stream offset of each window
    split:        str
    sigma_frozen: bool
    sigma_kind:   str                        # "diagonal" | "full" (the belief's covariance storage)
    diagnostics:  Dict[str, Any] = field(default_factory=dict)
    notes:        List[str] = field(default_factory=list)
    chart:        str = "local"              # "local" | "reference": coordinates of the cross-row statistics
    reference_variances: bool = False        # sigma/sigma0 hold the reference variances c (gaussian_frame_diagonal)
    xi:           Optional[np.ndarray] = None   # (L+1, n, K) float32 U_i^{-1} mu^(l)_i, reference chart only
    prior_xi:     Optional[np.ndarray] = None   # (n, K) float32 U_i^{-1} mu_{v_i}, reference chart only
    chart_sigma:  Optional[np.ndarray] = None   # (L+1, n, K) float32 chart covariance diagonal, None if frozen
    chart_sigma0: Optional[np.ndarray] = None   # (n, K) float32, None when the stored dispersion is the chart's
    kl:           Optional[np.ndarray] = None   # (L+1, n) float64 exact KL(q^(l)_i || q^(0)_i)

    @property
    def n(self) -> int:
        return int(self.token_ids.shape[0])

    @property
    def chart_mu(self) -> np.ndarray:
        r"""(L+1, n, K) means in the section's chart: ``xi`` under ``"reference"``, else the local ``mu``."""
        return self.xi if self.chart == "reference" else self.mu

    def chart_var(self, layer: int) -> np.ndarray:
        r"""(n, K) covariance diagonal of ``layer`` in the section's chart."""
        if self.chart_sigma0 is not None:
            return self.chart_sigma0 if self.chart_sigma is None else self.chart_sigma[layer]
        return self.sigma0 if self.sigma is None else self.sigma[layer]

    @property
    def n_layers(self) -> int:
        return int(self.mu.shape[0]) - 1

    @property
    def seq_len(self) -> int:
        return int(self.windows.shape[1])

    @property
    def K(self) -> int:
        return int(self.mu.shape[2])


def collect_bank(
    run:        Any,
    model:      Any,
    ctx:        AnalysisContext,

    *,
    split:      str = "validation",
    n_tokens:   int = 32768,
    seq_len:    Optional[int] = None,
    batch_size: int = 8,
) -> ContextualBank:
    r"""Per-layer beliefs of ``ceil(n_tokens / N)`` non-overlapping windows of ``split``.

    The stream (``run.token_stream(split)``, identity-checked for the checkpoint's selection split)
    is cut into consecutive slots of ``N + 1`` tokens (``N`` inputs plus the last target); a seeded
    random subset of slots, in random order, becomes the windows, so every position has its next
    token as target. Each batch runs ``model.forward_beliefs(x, capture={"diagnostic": {}})`` under
    ``torch.no_grad()``, halved by :func:`_forward_split` when the model's exact-symmetry check fails;
    windows that fail alone are dropped, and both counts are in ``diagnostics["forward_resplit"]``.
    Whether Sigma and phi move is measured by ``torch.equal`` against the stack input at every layer
    of every batch and recorded, together with the config-level flags. ``diagnostics["selection_split"]``
    is the split that selected the checkpoint (``None`` without a recorded identity), so a report can
    tell a held-out bank from one drawn on the selection split (audit 2026-10-16 F121).

    The chart arrays (``xi``, ``prior_xi``, ``chart_sigma``) and the exact KL (``kl``) are formed on the
    device from the realized frame of the same forward (:func:`_realized_frame`), see
    :class:`ContextualBank`. ``diagnostics["stack_input_equals_token_prior"]`` compares layer 0 with the
    token-table prior of each position's token (:func:`_token_prior_gap`).
    """
    cfg = run.cfg
    N = int(seq_len if seq_len is not None else cfg.max_seq_len)
    if not 2 <= N <= int(cfg.max_seq_len):
        raise ValueError(f"seq_len={N} must lie in [2, max_seq_len={cfg.max_seq_len}]")
    if n_tokens < 1 or batch_size < 1:
        raise ValueError("n_tokens and batch_size must be positive")
    t_start = time.perf_counter()
    tokens = _as_numpy_stream(run.token_stream(split))
    T = int(tokens.shape[0])
    S = N + 1
    n_slots = T // S
    if n_slots < 1:
        raise ValueError(f"the {split!r} stream has {T} tokens, fewer than one window of {S}")
    notes: List[str] = []
    n_windows = max(1, math.ceil(n_tokens / N))
    if n_windows > n_slots:
        notes.append(f"the {split!r} stream holds only {n_slots} disjoint windows of {S} tokens; the bank "
                     f"has {n_slots * N} tokens instead of the requested {n_tokens}")
        n_windows = n_slots
    slots = _rng(ctx, "bank").choice(n_slots, size=n_windows, replace=False)   # random order
    offsets = slots.astype(np.int64) * S
    full = np.stack([np.asarray(tokens[o:o + S], dtype=np.int64) for o in offsets])   # (n_seq, N+1)
    windows, targets = full[:, :N], full[:, 1:]
    if int(windows.min()) < 0 or int(full.max()) >= run.vocab_size:
        raise ValueError(f"the {split!r} stream has token ids outside [0, {run.vocab_size})")

    L, K = int(run.n_layers), int(run.K)
    n = n_windows * N
    mu = np.empty((L + 1, n, K), dtype=np.float32)
    var = np.empty((L + 1, n, K), dtype=np.float32)
    sigma_kind = "diagonal"
    chart = _section_chart(run)
    reference_variances = _stores_reference_variances(run)
    xi = np.empty((L + 1, n, K), dtype=np.float32) if chart == "reference" else None
    prior_xi = np.empty((n, K), dtype=np.float32) if chart == "reference" else None
    # The chart's covariance diagonal differs from the stored dispersion for the frame-diagonal family in local
    # coordinates (local diag of U diag(c) U^T) and for every other family in reference coordinates (diag C).
    chart_var = (np.empty((L + 1, n, K), dtype=np.float32)
                 if reference_variances != (chart == "reference") else None)
    kl: Optional[np.ndarray] = np.empty((L + 1, n), dtype=np.float64) if reference_variances else None
    device = torch.device(ctx.device)
    # Frozen-Σ/φ flags, accumulated on the device and read once at the end.
    sigma_flag = torch.ones((), dtype=torch.bool, device=device)
    phi_flag = torch.ones((), dtype=torch.bool, device=device)
    windows_dev = torch.from_numpy(windows).to(device)                 # (n_seq, N), one transfer
    split_log = _SplitLog()
    for b0 in range(0, n_windows, batch_size):
        b1 = min(n_windows, b0 + batch_size)
        x = windows_dev[b0:b1]                                          # (B, N)
        for s, e, out in _forward_split(model, x, split_log, offset=b0):
            layers = out["layers"]
            base = layers[0]
            if base.sigma.dim() == 4:
                sigma_kind = "full"
                if kl is None:                                          # the dense Gaussian KL
                    kl = np.empty((L + 1, n), dtype=np.float64)
            rows = slice((b0 + s) * N, (b0 + e) * N)
            # every layer of the piece stacked on the device, then one host transfer each
            mu[:, rows] = torch.stack([b.mu.reshape(-1, K) for b in layers]).float().cpu().numpy()   # (L+1, B*N, K)
            var[:, rows] = torch.stack([_covariance_diagonal(run, b.sigma).reshape(-1, K)
                                        for b in layers]).float().cpu().numpy()                     # (L+1, B*N, K)
            frame = _realized_frame(out) if (xi is not None or chart_var is not None or reference_variances) else None
            if xi is not None:
                xi[:, rows] = torch.stack([_reference_means(frame, b.mu).reshape(-1, K)
                                           for b in layers]).float().cpu().numpy()                  # (L+1, B*N, K)
                table = torch.from_numpy(np.asarray(run.mu(windows[b0 + s:b0 + e].reshape(-1)))).to(
                    device=base.mu.device, dtype=base.mu.dtype).reshape(base.mu.shape)               # (B, N, K) mu_v
                prior_xi[rows] = _reference_means(frame, table).reshape(-1, K).float().cpu().numpy()
            if chart_var is not None:
                chart_var[:, rows] = torch.stack([
                    _chart_covariance_diagonal(b, frame, reference_variances=reference_variances).reshape(-1, K)
                    for b in layers]).float().cpu().numpy()                                         # (L+1, B*N, K)
            if kl is not None:
                kl[:, rows] = _exact_layer_kl(layers, frame, reference_variances=reference_variances).cpu().numpy()
            for belief in layers[1:]:
                sigma_flag = sigma_flag & _equal_flag(belief.sigma, base.sigma, device)
                phi_flag = phi_flag & _equal_flag(belief.phi, base.phi, device)
    sigma_equal, phi_equal = bool(sigma_flag), bool(phi_flag)

    resplit = split_log.record()
    if split_log.skipped:                                               # drop windows that failed alone
        resplit["skipped_stream_offsets"] = [int(offsets[i]) for i in split_log.skipped]
        keep = np.ones(n_windows, dtype=bool)
        keep[split_log.skipped] = False
        if not keep.any():
            raise RuntimeError("every bank window failed the model's exact-symmetry check")
        rows_keep = np.repeat(keep, N)
        mu, var = mu[:, rows_keep], var[:, rows_keep]
        xi = None if xi is None else xi[:, rows_keep]
        prior_xi = None if prior_xi is None else prior_xi[rows_keep]
        chart_var = None if chart_var is None else chart_var[:, rows_keep]
        kl = None if kl is None else kl[:, rows_keep]
        windows, targets, offsets = windows[keep], targets[keep], offsets[keep]
        n_windows = int(keep.sum())
        n = n_windows * N
    prior_gap = _token_prior_gap(run, windows.reshape(-1), mu[0], var[0])
    seconds = time.perf_counter() - t_start
    if sigma_equal:
        notes.append("Σ is frozen in context: every layer's Σ equals the stack input's bitwise (torch.equal), "
                     "so contextual Σ analyses are vacuous and only μ carries context")
    if phi_equal:
        notes.append("φ is not updated by inference: every layer's φ equals the token frame; the realized "
                     "frame at position p is U = exp(X_v) exp(Y_p)")
    if run.sigma_frozen is not None and bool(run.sigma_frozen) != sigma_equal:
        notes.append(f"config-level covariance_frozen_at_encode={run.sigma_frozen} disagrees with the measured "
                     f"Σ-frozen={sigma_equal}; the measurement is used")
    if prior_gap["stack_input_equals_token_prior"] is False:
        rel_var = prior_gap["stack_input_minus_token_prior_max_rel_var"]
        notes.append(f"encode_mode={cfg.encode_mode!r}: layer 0 is the stack input, which differs from the "
                     f"token-table prior N(mu_v, Sigma_v) (max |Δμ| = "
                     f"{prior_gap['stack_input_minus_token_prior_max_abs_mu']:.3g}"
                     + (f", max relative Δvar = {rel_var:.3g}" if rel_var is not None else "") + ")")
    if chart == "reference":
        notes.append("decode_coordinates='reference': the decoder compares the reference means ξ = U⁻¹μ of the "
                     "realized frame with its decode rows, so the cross-row statistics (geometry, CKA, the cosine "
                     "terms of contextuality, the occurrence vectors) and the variance shares read ξ; the local "
                     "means depend on the trivialization there")
    if reference_variances:
        notes.append("gaussian_frame_diagonal stores the reference variances c (Σ_i = U_i diag(c_i) U_iᵀ): the KL "
                     "to the stack input is evaluated on (ξ = U⁻¹μ, c) of the realized frame, which is the family's "
                     "KL, and the variance shares read the covariance diagonal of the section's chart")
    return ContextualBank(
        token_ids=windows.reshape(-1).copy(), targets=targets.reshape(-1).copy(),
        positions=np.tile(np.arange(N, dtype=np.int64), n_windows),
        seq_index=np.repeat(np.arange(n_windows, dtype=np.int64), N),
        mu=mu, sigma=None if sigma_equal else var, phi_moves=not phi_equal, sigma0=var[0].copy(),
        windows=windows, offsets=offsets, split=split, sigma_frozen=sigma_equal, sigma_kind=sigma_kind,
        diagnostics={
            "split": split, "selection_split": (getattr(run, "selection_identity", None) or {}).get("split"),
            "seq_len": N, "n_windows": int(n_windows), "n_tokens": int(n),
            "n_tokens_requested": int(n_tokens), "n_slots": int(n_slots), "stream_tokens": T,
            "sigma_frozen_measured": sigma_equal, "sigma_frozen_config": run.sigma_frozen,
            "phi_moves_measured": not phi_equal, "phi_frozen_config": run.phi_frozen,
            **prior_gap, "sigma_kind": sigma_kind, "chart": chart,
            "reference_variances": reference_variances, "kl_exact": kl is not None,
            "forward_resplit": resplit,
            "seconds": seconds, "tokens_per_second": n / seconds if seconds > 0 else None,
        },
        notes=notes,
        chart=chart, reference_variances=reference_variances, xi=xi, prior_xi=prior_xi,
        chart_sigma=None if (chart_var is None or sigma_equal) else chart_var,
        chart_sigma0=None if chart_var is None else chart_var[0].copy(), kl=kl,
    )


# ----------------------------------------------------------------------------------------------
# Logit lens
# ----------------------------------------------------------------------------------------------

def logit_lens(
    run:          Any,
    model:        Any,
    bank:         ContextualBank,
    ctx:          AnalysisContext,

    *,
    max_tokens:   int = 8192,
    chunk_tokens: int = 1024,
) -> Dict[str, Any]:
    r"""Decode every layer's belief with the model's own final readout (the logit lens).

    For the first ``floor(max_tokens / N)`` bank windows (a random subset: bank windows are in random
    order) the windows are re-run and each layer's belief :math:`q^{(l)}_i` is read out by
    :func:`_readout`. With :math:`p^{(l)}_i = \mathrm{softmax}
    \ell^{(l)}_i` and the next token :math:`y_i`, per layer:

    .. math::
        \mathrm{CE}_l = -\log p^{(l)}_i(y_i), \quad
        \mathrm{KL}(p^{(L)}_i \| p^{(l)}_i) = \sum_v p^{(L)}_{iv} (\log p^{(L)}_{iv} - \log p^{(l)}_{iv}), \quad
        H_l = -\sum_v p^{(l)}_{iv} \log p^{(l)}_{iv},

    top-1 accuracy and top-1 agreement with the final layer, each as a mean with a 95% cluster
    (window) bootstrap CI. ``consistency`` records the largest difference between the last-layer lens
    logits and the forward's own logits (the lens contract; zero when the readout is replicated
    exactly), measured on the first chunk that decodes (``windows_checked`` windows): only that chunk's
    forward is run with ``return_logits=True``, so every other chunk decodes its final layer once, and
    the forward's logits are released right after the check. It also records the largest difference
    between the re-run beliefs and the bank over every chunk. The unigram readout
    :math:`\mathrm{softmax}(\kappa\log\pi)` gives the context-free baseline CE. Forwards go through
    :func:`_forward_split`; ``window_index`` lists the bank windows actually decoded and
    ``forward_resplit`` the re-split and skipped counts.

    Every statistic of a chunk is reduced on the model's device; the chunk's ``(L+1, 5, n_valid)``
    statistics and its re-run means go to the host in one transfer each. The row sums
    :math:`\sum_v p_v \log p_v` are batched matmuls (:func:`_rowdot`), so a chunk holds a few
    ``(B, N, V)`` tensors at a time (``chunk_tokens`` bounds ``B N``).
    """
    N = bank.seq_len
    n_win = min(bank.windows.shape[0], max(1, int(max_tokens) // N))
    B = max(1, int(chunk_tokens) // N)
    L = bank.n_layers
    targets_w = bank.targets.reshape(-1, N)                              # (n_seq, N)
    device = torch.device(ctx.device)
    keys = ("ce", "acc", "agree", "kl", "entropy")
    per: Dict[str, List[List[np.ndarray]]] = {k: [[] for _ in range(L + 1)] for k in keys}
    clusters: List[np.ndarray] = []
    pos_kept: List[np.ndarray] = []
    target_kept: List[np.ndarray] = []
    logit_diff = torch.zeros((), dtype=torch.float32, device=device)   # read once at the end
    windows_checked = 0                                                  # windows of the forward-logit contract check
    max_mu_diff = 0.0
    example: Optional[Dict[str, Any]] = None
    P = min(N, 16)
    split_log = _SplitLog()
    used: List[int] = []                                                 # bank windows actually decoded
    windows_dev = torch.from_numpy(bank.windows[:n_win]).to(device)    # (n_win, N), one transfer
    targets_dev = torch.from_numpy(targets_w[:n_win]).to(device)       # (n_win, N)
    t_start = time.perf_counter()
    for c0 in range(0, n_win, B):
        x = windows_dev[c0:min(n_win, c0 + B)]                          # (B, N)
        check = windows_checked == 0                                     # forward logits until one chunk decoded
        for s, e, out in _forward_split(model, x, split_log, return_logits=check, offset=c0):
            b0, b1 = c0 + s, c0 + e
            used.extend(range(b0, b1))
            layers, frame = out["layers"], out["canonical_frame"]
            tgt = targets_dev[b0:b1]                                    # (B, N)
            valid = tgt >= 0
            safe_tgt = tgt.clamp_min(0)[..., None]                      # (B, N, 1)
            rerun = torch.stack([b.mu.reshape(-1, bank.K) for b in layers]).float().cpu().numpy()   # (L+1, B*N, K)
            max_mu_diff = max(max_mu_diff, float(np.abs(rerun - bank.mu[:, b0 * N:b1 * N]).max()))
            logits_L = _readout(model, layers[L].mu, layers[L].sigma, frame,
                                encoded_sigma=layers[0].sigma)                  # (B, N, V)
            if out["logits"] is not None:                               # the lens contract, then release them
                logit_diff = torch.maximum(logit_diff, (logits_L - out["logits"]).abs().max().float())
                windows_checked += b1 - b0
                out["logits"] = None
            top_L = logits_L.argmax(dim=-1)                             # (B, N)
            logp_L = torch.log_softmax(logits_L.float(), dim=-1)        # (B, N, V)
            del logits_L
            p_L = logp_L.exp()                                          # (B, N, V)
            stats_l: List[torch.Tensor] = []
            top_first: List[torch.Tensor] = []
            for l in range(L + 1):
                if l == L:
                    logp, top = logp_L, top_L
                else:
                    logits = _readout(model, layers[l].mu, layers[l].sigma, frame,
                                      encoded_sigma=layers[0].sigma)           # (B, N, V)
                    top = logits.argmax(dim=-1)                          # (B, N)
                    logp = torch.log_softmax(logits.float(), dim=-1)     # (B, N, V)
                    del logits
                ce = -logp.gather(-1, safe_tgt)[..., 0]                  # (B, N)
                entropy = -_rowdot(logp.exp(), logp)                     # (B, N)
                kl = _rowdot(p_L, logp_L - logp)                         # KL(p_L || p_l), (B, N)
                stats_l.append(torch.stack([ce.double(), (top == tgt).double(), (top == top_L).double(),
                                            kl.double(), entropy.double()]))   # (5, B, N)
                if example is None:
                    top_first.append(top[0, :P])
                del logp
            piece = torch.stack(stats_l)[:, :, valid].cpu().numpy()     # (L+1, 5, n_valid), one transfer
            for l in range(L + 1):
                for k, key in enumerate(keys):
                    per[key][l].append(piece[l, k])
            if example is None:
                example = {"window": bank.windows[b0, :P].copy(), "targets": targets_w[b0, :P].copy(),
                           "top1": torch.stack(top_first).cpu().numpy()}   # (L+1, P)
            del logp_L, p_L
            seq = np.repeat(np.arange(b0, b1, dtype=np.int64), N).reshape(b1 - b0, N)
            keep = targets_w[b0:b1] >= 0                                 # the host copy of ``valid``
            clusters.append(seq[keep])
            pos_kept.append(np.tile(np.arange(N, dtype=np.int64), (b1 - b0, 1))[keep])
            target_kept.append(targets_w[b0:b1][keep])

    if not clusters:
        raise RuntimeError("every logit-lens window failed the model's exact-symmetry check")
    max_logit_diff = float(logit_diff) if windows_checked else None
    cl = np.concatenate(clusters)
    positions = np.concatenate(pos_kept)
    tgt_all = np.concatenate(target_kept)
    rng = _rng(ctx, "lens")
    n_boot = int(_opt(ctx, "n_boot"))
    arrays = {k: [np.concatenate(v) for v in per[k]] for k in per}
    result: Dict[str, Any] = {
        "n_tokens": int(cl.size), "n_windows": len(used), "n_windows_requested": int(n_win), "seq_len": N,
        "layers": list(range(L + 1)), "layer_names": _layer_names(L, run.cfg.encode_mode),
        "decode_mode": run.cfg.decode_mode,
        "forward_resplit": split_log.record(), "notes": [],
    }
    for key in ("ce", "acc", "agree", "kl", "entropy"):
        result[key] = [_cluster_mean_ci(arrays[key][l], cl, rng, n_boot) for l in range(L + 1)]
    result["kl_quantiles"] = [_quantiles(arrays["kl"][l]) for l in range(L + 1)]
    counts = np.bincount(positions, minlength=N).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        result["ce_by_position"] = np.stack([
            np.bincount(positions, weights=arrays["ce"][l], minlength=N) / counts for l in range(L + 1)])
    agree_means = [r["mean"] for r in result["agree"]]
    result["first_layer_agree_half"] = next((l for l, a in enumerate(agree_means) if a >= 0.5), None)
    result["consistency"] = {"max_abs_logit_diff_final_vs_forward": max_logit_diff,
                             "windows_checked": int(windows_checked),
                             "max_abs_mu_diff_rerun_vs_bank": max_mu_diff}
    u = run.unigram_log_prior()
    if u is not None and bool(run.cfg.decode_unigram_prior):
        kappa = float(run.cfg.unigram_kappa)
        z = kappa * u
        log_q = z - (np.log(np.exp(z - z.max()).sum()) + z.max())      # log softmax(kappa log pi)
        result["unigram_ce"] = _cluster_mean_ci(-log_q[tgt_all], cl, rng, n_boot)
    else:
        result["unigram_ce"] = None
        result["notes"].append("no unigram table in the decode: the context-free unigram CE baseline is "
                               "not available")
    result["example"] = example
    result["arrays"] = arrays                                            # per-token values (dropped from metrics)
    result["clusters"] = cl
    result["window_index"] = np.asarray(used, dtype=np.int64)          # bank windows decoded, ascending
    result["seconds"] = time.perf_counter() - t_start
    result["notes"].append("intermediate layers are decoded with the final readout (logit lens); the blocks "
                           "were never trained to be read out there, so the early-layer numbers describe what "
                           "the final readout makes of those beliefs, not all the information they carry (no "
                           "tuned lens is fitted)")
    return result


# ----------------------------------------------------------------------------------------------
# Per-layer geometry and CKA
# ----------------------------------------------------------------------------------------------

def layer_geometry(bank: ContextualBank, ctx: AnalysisContext) -> Dict[str, Any]:
    r"""Global geometry of the contextual means at every layer on one row-matched token sample.

    Per layer: mean cosine (raw and centered; Ethayarajh 2019), mean-vector ratio, IsoScore (Rudman
    et al. 2022), effective rank (Roy & Vetterli 2007), participation ratio, the top-10 explained
    variance ratios, TwoNN intrinsic dimension (Facco et al. 2017), rogue-dimension share (Timkey &
    van Schijndel 2021) and the median norm. Layer 0 rows are the stack input, i.e. the token priors
    of the sampled positions (duplicates of repeated tokens are dropped by TwoNN). The rows are the
    means in the section's chart (``bank.chart_mu``: :math:`\xi = U^{-1}\mu` under a reference-coordinate
    decoder), recorded as ``chart``.
    """
    rows = _token_sample(bank, ctx, int(_opt(ctx, "contextual_points")))
    rng = _rng(ctx, "geometry")
    out: Dict[str, Any] = {"n": int(rows.size), "K": bank.K, "chart": bank.chart, "layers": [], "notes": []}
    means = bank.chart_mu                                                # (L+1, n, K)
    for l in range(bank.n_layers + 1):
        X = means[l, rows].astype(np.float64)                           # (n, K)
        rec: Dict[str, Any] = {"layer": l}
        notes = out["notes"]
        _attempt(rec, "anisotropy_raw", lambda: stats.anisotropy(X), notes)
        _attempt(rec, "anisotropy_centered", lambda: stats.anisotropy(X, center=True), notes)
        _attempt(rec, "mean_vector_ratio", lambda: stats.mean_vector_ratio(X), notes)
        _attempt(rec, "isoscore", lambda: stats.isoscore(X), notes)
        _attempt(rec, "effective_rank", lambda: stats.effective_rank(X), notes)
        _attempt(rec, "participation_ratio", lambda: stats.participation_ratio(X), notes)
        _attempt(rec, "pca_top10", lambda: stats.pca_spectrum(X, k=10), notes)
        _attempt(rec, "twonn", lambda: stats.twonn_dimension(X, rng, max_points=rows.size), notes)
        _attempt(rec, "rogue", lambda: stats.rogue_dimensions(X, top=3), notes)
        rec["norm_median"] = float(np.median(np.linalg.norm(X, axis=1)))
        out["layers"].append(rec)
    return out


def layer_cka(bank: ContextualBank, run: Any, ctx: AnalysisContext) -> Dict[str, Any]:
    r"""Linear CKA between layers, with the token prior mean, and with the next token's decode row.

    On one row-matched token sample: the ``(L+1) x (L+1)`` matrices of biased and debiased linear CKA
    (Kornblith et al. 2019; unbiased HSIC of Song et al. 2012) between the contextual means; per layer
    :math:`\mathrm{CKA}(\mu^{(l)}, \mu_{v_i})` with the static prior mean of the position's token, and
    :math:`\mathrm{CKA}(\mu^{(l)}, w_{y_i})` with the decode row of the NEXT token (linear decode:
    ``output_proj_weight``; Gaussian decode: the decode means :math:`m_v`) -- how "output-like"
    each layer is. CKA is invariant to orthogonal maps and isotropic scaling, not to the global
    positive-diagonal rescaling of the coordinates. The contextual and prior means are those of the
    section's chart (``bank.chart_mu``, ``chart``): under a reference-coordinate decoder
    :math:`\xi^{(l)}_i = U_i^{-1}\mu^{(l)}_i` and :math:`U_i^{-1}\mu_{v_i}`, the coordinates the decoder
    compares with its decode rows.
    """
    rows = _token_sample(bank, ctx, int(_opt(ctx, "contextual_points")))
    L = bank.n_layers
    means = bank.chart_mu                                                # (L+1, n, K)
    Xs = [means[l, rows].astype(np.float64) for l in range(L + 1)]
    biased = np.eye(L + 1)
    debiased = np.eye(L + 1)
    for a in range(L + 1):
        for b in range(a + 1, L + 1):
            biased[a, b] = biased[b, a] = stats.linear_cka(Xs[a], Xs[b])
            debiased[a, b] = debiased[b, a] = stats.linear_cka(Xs[a], Xs[b], debiased=True)
    if bank.chart == "reference":
        prior = bank.prior_xi[rows].astype(np.float64)                   # (n, K) U_i^{-1} mu_{v_i}
    else:
        prior = run.mu(bank.token_ids[rows])                             # (n, K)
    out: Dict[str, Any] = {
        "n": int(rows.size), "chart": bank.chart, "biased": biased, "debiased": debiased,
        "prior_mu": {"biased": [stats.linear_cka(X, prior) for X in Xs],
                     "debiased": [stats.linear_cka(X, prior, debiased=True) for X in Xs]},
        "notes": [],
    }
    tgt = bank.targets[rows]
    valid = tgt >= 0
    if run.decode_kind == "linear":
        decode_rows, label = run.decode_W(tgt[valid]), "linear decode row w_y (output_proj_weight)"
    elif run.decode_kind in GAUSSIAN_BANK_DECODE_KINDS:
        decode_rows, label = run.decode_mu(tgt[valid]), "Gaussian decode mean m_y"
    else:
        decode_rows, label = None, None
        out["notes"].append(f"decode kind {run.decode_kind!r}: no decode rows for the next-token CKA")
    if decode_rows is not None and int(valid.sum()) >= 4:
        out["next_token_decode"] = {
            "label": label, "n": int(valid.sum()),
            "biased": [stats.linear_cka(X[valid], decode_rows) for X in Xs],
            "debiased": [stats.linear_cka(X[valid], decode_rows, debiased=True) for X in Xs]}
    else:
        out["next_token_decode"] = None
    return out


# ----------------------------------------------------------------------------------------------
# Contextuality
# ----------------------------------------------------------------------------------------------

def _mev(X: np.ndarray) -> float:
    r"""Maximum explainable variance :math:`\sigma_1^2 / \sum_i \sigma_i^2` of an (uncentered) occurrence matrix."""
    s = np.linalg.svd(X, compute_uv=False)
    total = float((s * s).sum())
    return float(s[0] ** 2 / total) if total > 0.0 else float("nan")


def _kl_to_stack_input(bank: ContextualBank, layer: int) -> np.ndarray:
    r"""(n,) :math:`\mathrm{KL}(q^{(l)}_i \| q^{(0)}_i)` of every bank row in the family's geometry.

    ``bank.kl`` where the bank holds the exact values (:func:`_exact_layer_kl`: the frame-diagonal family, a full
    covariance; NaN where a covariance has no Cholesky factor). Otherwise the diagonal Gaussians of the stored local
    means and covariance diagonals, which are the family's own: with Sigma frozen the Mahalanobis displacement
    :math:`\tfrac12\sum_k \Delta\mu_k^2/\sigma^{(0)}_k`, else the diagonal-Gaussian KL.
    """
    from vfe3.semantics.gaussians import kl_diag

    if bank.kl is not None:
        return np.asarray(bank.kl[layer], dtype=np.float64)
    M = bank.mu[layer].astype(np.float64)                                # (n, K)
    mu0 = bank.mu[0].astype(np.float64)
    var0 = bank.sigma0.astype(np.float64)
    if bank.sigma is None:
        dmu = M - mu0
        return 0.5 * (dmu * dmu / var0).sum(axis=1)
    return kl_diag(M, np.log(bank.sigma[layer].astype(np.float64)), mu0, np.log(var0))


def _kl_form(bank: ContextualBank) -> str:
    r"""The label of :func:`_kl_to_stack_input` for this bank (frozen or moving Sigma, and where it is evaluated)."""
    if bank.sigma is None:
        form = "Mahalanobis displacement ½ Δμᵀ Σ₀⁻¹ Δμ (Σ frozen)"
        if bank.kl is not None:
            form += (", evaluated as ½ Σ_k Δξ_k²/c_k on ξ = U⁻¹μ" if bank.reference_variances
                     else ", with the dense Σ₀")
        return form
    if bank.kl is None:
        return "diagonal-Gaussian KL(q_l ‖ q_0)"
    return ("Gaussian KL(q_l ‖ q_0), evaluated on (ξ = U⁻¹μ, c)" if bank.reference_variances
            else "dense Gaussian KL(q_l ‖ q_0)")


def contextuality(bank: ContextualBank, run: Any, ctx: AnalysisContext) -> Dict[str, Any]:
    r"""How context-specific the contextual means are, per layer (Ethayarajh 2019), plus Gaussian terms.

    Over token types with at least ``min_type_occurrences`` bank occurrences (at most
    ``max_type_occurrences`` used, seeded), with unit rows :math:`z` of the means at layer ``l`` in the
    section's chart (``bank.chart_mu``: :math:`\xi = U^{-1}\mu` under a reference-coordinate decoder,
    else the local means):

    * self-similarity :math:`\mathrm{SelfSim}(w) = \frac{1}{n(n-1)}\sum_{j \ne k} z_j^\top z_k
      = \frac{\lVert\sum_j z_j\rVert^2 - n}{n(n-1)}`;
    * intra-window similarity :math:`\mathrm{IntraSim}(s) = \frac1N \sum_i \cos(\mu_i, \bar\mu_s)`;
    * maximum explainable variance :math:`\mathrm{MEV}(w) = \sigma_1^2/\sum_i\sigma_i^2` of the
      occurrence matrix;

    each also minus its random baseline (the mean cosine of random bank positions at that layer;
    for MEV, the MEV of random position sets of the same size). Gaussian terms, with :math:`q^{(0)}_i`
    the stack input of the same position (the token prior for ``per_token`` encode):

    * :math:`\mathrm{KL}(q^{(l)}_i \| q^{(0)}_i)` in the family's geometry (:func:`_kl_to_stack_input`),
      which with Sigma frozen is the Mahalanobis displacement
      :math:`\tfrac12 \Delta\mu^\top \Sigma_{0}^{-1}\Delta\mu` (chart-free, so invariant under the
      global positive-diagonal rescaling and every frame change);
    * the within-type variance share :math:`\rho_l(w) = \mathrm{tr} \mathrm{Cov}_c(\mu_c) /
      (\mathrm{tr} \mathrm{Cov}_c(\mu_c) + \mathrm{tr} \bar\Sigma)`, :math:`\bar\Sigma = E_c\Sigma_c`
      (the law-of-total-covariance split of the moment-matched pooled Gaussian), and its per-coordinate
      mean :math:`\frac1K\sum_k \mathrm{Var}_c(\mu_{ck})/(\mathrm{Var}_c(\mu_{ck}) + \bar\Sigma_{kk})`,
      which unlike the trace ratio is invariant under the diagonal rescaling. Means and covariance
      diagonals are both those of the section's chart (``bank.chart_mu``, ``bank.chart_var``).
    """
    rng = _rng(ctx, "contextuality")
    n_boot = int(_opt(ctx, "n_boot"))
    min_occ = int(_opt(ctx, "min_type_occurrences"))
    max_occ = int(_opt(ctx, "max_type_occurrences"))
    if min_occ < 2:
        raise ValueError("min_type_occurrences must be at least 2")
    types, inverse, counts = np.unique(bank.token_ids, return_inverse=True, return_counts=True)
    eligible = np.flatnonzero(counts >= min_occ)
    out: Dict[str, Any] = {"min_occurrences": min_occ, "max_occurrences": max_occ,
                           "n_types": int(eligible.size), "layers": [], "notes": []}
    if eligible.size < 2:
        out.update(status="skipped", reason=f"{eligible.size} token type(s) with >= {min_occ} occurrences")
        return out
    order = np.argsort(inverse, kind="stable")                          # rows grouped by type
    starts = np.r_[0, np.cumsum(counts)]
    groups: List[np.ndarray] = []
    for t in eligible:
        members = order[starts[t]:starts[t + 1]]
        if members.size > max_occ:
            members = np.sort(rng.choice(members, size=max_occ, replace=False))
        groups.append(members)
    sizes = np.array([g.size for g in groups], dtype=np.int64)
    flat = np.concatenate(groups)                                       # (M,)
    offsets = np.r_[0, np.cumsum(sizes)[:-1]]
    type_ids = types[eligible]
    codes, class_names = run.vocab.token_classes()
    type_class = codes[type_ids]
    sample = _token_sample(bank, ctx, int(_opt(ctx, "contextual_points")))
    N = bank.seq_len
    var0 = bank.sigma0.astype(np.float64)                               # (n, K)
    if np.any(var0 <= 0.0):
        raise ValueError("stack-input covariance diagonal has non-positive entries")
    out["chart"] = bank.chart
    out["kl_form"] = _kl_form(bank)
    if bank.sigma_kind == "full":
        out["notes"].append("Σ is full: the KL to the stack input uses the dense covariances, and the variance "
                            "shares read its diagonal, which determines the trace ratio exactly")
    if not str(run.cfg.family).startswith("gaussian"):
        out["notes"].append(f"family {run.cfg.family}: KL terms are those of the moment-matched diagonal Gaussians")
    means = bank.chart_mu                                                # (L+1, n, K) chart means
    mev_baseline: Dict[int, List[float]] = {}
    most_least: Dict[str, Any] = {}
    n_kl_dropped = 0
    for l in range(bank.n_layers + 1):
        M = means[l].astype(np.float64)                                 # (n, K)
        norms = np.linalg.norm(M, axis=1)
        if np.any(norms <= 0.0):
            raise ValueError(f"layer {l} has zero-norm contextual means: cosine is undefined")
        Z = M / norms[:, None]
        baseline = stats.anisotropy(M[sample])
        sums = np.add.reduceat(Z[flat], offsets, axis=0)               # (T, K)
        selfsim = ((sums * sums).sum(axis=1) - sizes) / (sizes * (sizes - 1.0))
        mev = np.array([_mev(M[g]) for g in groups])
        for m in np.unique(sizes):
            if int(m) not in mev_baseline:
                mev_baseline[int(m)] = []
            if len(mev_baseline[int(m)]) == l:
                draws = [_mev(M[rng.choice(bank.n, size=int(m), replace=False)]) for _ in range(20)]
                mev_baseline[int(m)].append(float(np.mean(draws)))
        mev_adj = mev - np.array([mev_baseline[int(m)][l] for m in sizes])
        W = M.reshape(-1, N, bank.K)
        mean_w = W.mean(axis=1)                                          # (n_seq, K)
        mean_norm = np.linalg.norm(mean_w, axis=1, keepdims=True)
        if np.any(mean_norm <= 0.0):
            raise ValueError(f"layer {l}: a window has a zero mean vector, IntraSim is undefined")
        intra = (Z.reshape(-1, N, bank.K) * (mean_w / mean_norm)[:, None, :]).sum(axis=2).mean(axis=1)
        var_l = bank.chart_var(l).astype(np.float64)                    # (n, K) chart covariance diagonal
        kl = _kl_to_stack_input(bank, l)                                 # (n,)
        finite = np.isfinite(kl)
        n_kl_dropped = max(n_kl_dropped, int((~finite).sum()))
        kl_rows, kl_windows = (kl, bank.seq_index) if finite.all() else (kl[finite], bank.seq_index[finite])
        share, share_pc = [], []
        for g in groups:
            cov_diag = M[g].var(axis=0)                                  # Var_c(mu_ck), (K,)
            sig_bar = var_l[g].mean(axis=0)                              # E_c Sigma_c diagonal, (K,)
            share.append(cov_diag.sum() / (cov_diag.sum() + sig_bar.sum()))
            share_pc.append(float(np.mean(cov_diag / (cov_diag + sig_bar))))
        adj = selfsim - baseline
        by_class = {}
        for c, name in enumerate(class_names):
            mask = type_class == c
            if mask.any():
                by_class[name] = {"n_types": int(mask.sum()), "self_similarity_adjusted": float(adj[mask].mean())}
        out["layers"].append({
            "layer": l, "baseline_cosine": baseline,
            "self_similarity": stats.bootstrap_mean_ci(selfsim, rng, n_boot),
            "self_similarity_adjusted": stats.bootstrap_mean_ci(adj, rng, n_boot),
            "intra_similarity": stats.bootstrap_mean_ci(intra, rng, n_boot),
            "intra_similarity_adjusted": stats.bootstrap_mean_ci(intra - baseline, rng, n_boot),
            "mev": stats.bootstrap_mean_ci(mev, rng, n_boot),
            "mev_adjusted": stats.bootstrap_mean_ci(mev_adj, rng, n_boot),
            "kl_to_stack_input": {**_cluster_mean_ci(kl_rows, kl_windows, rng, n_boot), **_quantiles(kl_rows)},
            "variance_share": stats.bootstrap_mean_ci(np.asarray(share), rng, n_boot),
            "variance_share_per_coordinate": stats.bootstrap_mean_ci(np.asarray(share_pc), rng, n_boot),
            "by_class": by_class,
        })
        if l == bank.n_layers:
            rank = np.argsort(adj, kind="stable")
            most_least = {
                "most_contextual": [{"token": int(type_ids[i]), "n": int(sizes[i]),
                                     "self_similarity_adjusted": float(adj[i])} for i in rank[:10]],
                "least_contextual": [{"token": int(type_ids[i]), "n": int(sizes[i]),
                                      "self_similarity_adjusted": float(adj[i])} for i in rank[::-1][:10]],
            }
    out["n_windows"] = int(bank.windows.shape[0])
    out["mev_baseline"] = {str(m): v for m, v in sorted(mev_baseline.items())}
    out["final_layer_extremes"] = most_least
    if n_kl_dropped:
        out["notes"].append(f"up to {n_kl_dropped} positions per layer whose covariance has no Cholesky factor are "
                            "left out of the KL to the stack input")
    status, reason = diagonal_redundancy_status(run.cfg)
    out["diagonal_redundancy"] = {"status": status, "reason": reason}
    if bank.chart == "reference":
        out["notes"].append("cosine-based terms and variance shares use the reference means ξ = U⁻¹μ that the "
                            "decoder compares with its decode rows; the per-position frame action U_i → g_i U_i "
                            "and the global positive-diagonal rescaling leave ξ unchanged, a global right action "
                            "U → U h maps it to h⁻¹ξ, and the KL terms are chart-free")
    elif status == "broken":
        out["notes"].append("cosine-based terms use the stored coordinates of μ; the global positive-diagonal "
                            f"rescaling is not a redundancy of this run ({reason}), so these coordinates are "
                            "not a gauge choice")
    elif status == "exact":
        out["notes"].append("cosine-based terms use the stored coordinates of μ; they are not invariant under the "
                            "global positive-diagonal rescaling, an exact redundancy of this run, while the KL "
                            "terms are")
    else:
        out["notes"].append("cosine-based terms use the stored coordinates of μ; they are not invariant under the "
                            "global positive-diagonal rescaling, which is assumed but not verified to be a "
                            f"redundancy of this run ({reason}), while the KL terms are")
    return out


# ----------------------------------------------------------------------------------------------
# Occurrence contexts (training split) and layerwise static distillation
# ----------------------------------------------------------------------------------------------

def _count_eligible(tokens: np.ndarray, code: np.ndarray, n_words: int, min_pos: int,
                    chunk: int) -> np.ndarray:
    r"""Occurrences at positions ``p >= min_pos`` of each coded word, by chunked lookup."""
    counts = np.zeros(n_words, dtype=np.int64)
    for s in range(min_pos, tokens.shape[0], chunk):
        c = code[np.asarray(tokens[s:s + chunk])]
        counts += np.bincount(c[c >= 0], minlength=n_words)
    return counts


def _positions_of_ranks(tokens: np.ndarray, code: np.ndarray, chosen_keys: np.ndarray, n_words: int,
                        min_pos: int, chunk: int) -> Tuple[np.ndarray, np.ndarray]:
    r"""Stream positions of the chosen occurrence ranks (key = word code * 2**40 + rank)."""
    seen = np.zeros(n_words, dtype=np.int64)
    out_pos: List[np.ndarray] = []
    out_code: List[np.ndarray] = []
    for s in range(min_pos, tokens.shape[0], chunk):
        c_all = code[np.asarray(tokens[s:s + chunk])]
        hit = np.flatnonzero(c_all >= 0)
        if hit.size == 0:
            continue
        c = c_all[hit]
        order = np.argsort(c, kind="stable")                            # hits grouped by word, in stream order
        cs = c[order]
        first = np.r_[0, np.flatnonzero(np.diff(cs)) + 1]
        group_start = np.repeat(first, np.diff(np.r_[first, cs.size]))
        rank = seen[cs] + (np.arange(cs.size) - group_start)
        keep = np.isin(cs * (1 << 40) + rank, chosen_keys)
        out_pos.append(hit[order][keep] + s)
        out_code.append(cs[keep])
        seen += np.bincount(c, minlength=n_words)
    if not out_pos:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(out_pos).astype(np.int64), np.concatenate(out_code).astype(np.int64)


def collect_occurrences(
    run:                Any,
    model:              Any,
    ctx:                AnalysisContext,
    word_ids:           Sequence[int],

    *,
    split:              str = "train",
    per_word:           int = 4,
    context_len:        int = 48,
    budget_tokens:      int = 131072,
    per_word_overrides: Optional[Mapping[int, int]] = None,
    batch_size:         int = 8,
    chunk:              int = 1 << 23,
) -> Dict[str, Any]:
    r"""Per-layer beliefs of sampled occurrences of ``word_ids``, each at the end of its context.

    Occurrences are located with numpy on the (memory-mapped) stream in chunks of ``chunk`` tokens.
    Only positions :math:`p \ge` ``context_len`` - 1 qualify, so every window
    ``stream[p - context_len + 1 : p + 1]`` has full left context and ends at the occurrence. The
    budget allows ``budget_tokens // context_len`` windows; words are served in the given (priority)
    order, each with ``per_word`` occurrences (``per_word_overrides[w]`` if given), fewer when the word
    is rarer or the budget runs out; the occurrences of a word are a seeded uniform sample without
    replacement of its qualifying positions. Returns per word the ``(L+1, n_occ, K)`` means at the last
    window position, the contexts, the positions, and the words that were missing from the split or
    dropped by the budget. Windows run ``batch_size`` at a time through :func:`_forward_split`; an
    occurrence whose window fails the model's exact-symmetry check alone is dropped and recorded in
    ``forward_resplit`` (with its stream position).

    The means are those of the section's chart (``chart``, :func:`_section_chart`): under a
    reference-coordinate decoder :math:`\xi = U^{-1}\mu` of the realized frame at the occurrence, and
    ``static_reference`` holds per word the token-table mean in the same chart, :math:`U^{-1}\mu_v` at
    the occurrence position (``None`` in the local chart, where the static table is ``run.mu``).
    """
    if context_len < 1 or per_word < 1 or budget_tokens < context_len or batch_size < 1:
        raise ValueError("context_len, per_word, batch_size and budget_tokens must allow at least one window")
    t_start = time.perf_counter()
    words = list(dict.fromkeys(int(w) for w in word_ids))
    V = int(run.vocab_size)
    if any(not 0 <= w < V for w in words):
        raise ValueError(f"word ids must lie in [0, {V})")
    requested = {w: int((per_word_overrides or {}).get(w, per_word)) for w in words}
    tokens = _as_numpy_stream(run.token_stream(split))
    min_pos = context_len - 1
    code = np.full(V, -1, dtype=np.int64)
    code[np.asarray(words, dtype=np.int64)] = np.arange(len(words))
    available = _count_eligible(tokens, code, len(words), min_pos, chunk)
    remaining = int(budget_tokens) // int(context_len)
    rng = _rng(ctx, "occurrences")
    allocated: Dict[int, int] = {}
    missing, dropped, partial = [], [], {}
    chosen: List[np.ndarray] = []
    for j, w in enumerate(words):
        if available[j] == 0:
            missing.append(w)
            continue
        if remaining == 0:
            dropped.append(w)
            continue
        k = int(min(requested[w], available[j], remaining))
        remaining -= k
        allocated[w] = k
        if k < requested[w]:
            partial[w] = k
        ranks = rng.choice(int(available[j]), size=k, replace=False)
        chosen.append(j * (1 << 40) + np.sort(ranks))
    notes: List[str] = []
    chart = _section_chart(run)
    result: Dict[str, Any] = {
        "split": split, "context_len": int(context_len), "per_word": int(per_word),
        "budget_tokens": int(budget_tokens), "requested": requested, "allocated": allocated, "missing": missing,
        "dropped_budget": dropped, "partial": partial, "notes": notes, "chart": chart,
    }
    if not chosen:
        result.update(word_ids=np.empty(0, dtype=np.int64), vectors={}, sigma=None, contexts={},
                      positions={}, n_occurrences={}, n_forward_tokens=0, n_layers=int(run.n_layers),
                      static_reference=None, seconds=time.perf_counter() - t_start)
        notes.append("no requested word occurs in the split with full left context")
        return result
    chosen_keys = np.sort(np.concatenate(chosen))
    pos, occ_code = _positions_of_ranks(tokens, code, chosen_keys, len(words), min_pos, chunk)
    if pos.size != chosen_keys.size:
        raise RuntimeError(f"located {pos.size} of {chosen_keys.size} chosen occurrences")
    # windows grouped by word in priority order, positions ascending within a word
    order = np.lexsort((pos, occ_code))
    pos, occ_code = pos[order], occ_code[order]
    offsets = np.arange(-context_len + 1, 1, dtype=np.int64)
    contexts = np.asarray(tokens[(pos[:, None] + offsets[None, :]).reshape(-1)], dtype=np.int64)
    contexts = contexts.reshape(pos.size, context_len)                   # (m, context_len)
    if not np.array_equal(contexts[:, -1], np.asarray(words, dtype=np.int64)[occ_code]):
        raise RuntimeError("an occurrence window does not end at its word")

    L, K = int(run.n_layers), int(run.K)
    m = pos.size
    mu = np.empty((L + 1, m, K), dtype=np.float32)
    var = np.empty((L + 1, m, K), dtype=np.float32)
    static = np.empty((m, K), dtype=np.float32) if chart == "reference" else None
    device = torch.device(ctx.device)
    sigma_flag = torch.ones((), dtype=torch.bool, device=device)      # Σ equal at every layer; read once
    contexts_dev = torch.from_numpy(contexts).to(device)               # (m, context_len), one transfer
    split_log = _SplitLog()
    for c0 in range(0, m, batch_size):
        x = contexts_dev[c0:min(m, c0 + batch_size)]                    # (B, context_len)
        for s, e, out in _forward_split(model, x, split_log, offset=c0):
            b0, b1 = c0 + s, c0 + e
            layers = out["layers"]
            base = layers[0]
            # the last position of every layer stacked on the device, then one host transfer each
            if static is not None:                                      # xi = U^{-1} mu at the occurrence
                frame = _realized_frame(out)
                last = SimpleNamespace(inverse=frame.inverse[:, -1:], block_diagonal=frame.block_diagonal)
                mu[:, b0:b1] = torch.stack([_reference_means(last, b.mu[:, -1:])[:, 0]
                                            for b in layers]).float().cpu().numpy()                  # (L+1, B, K)
                table = torch.from_numpy(np.asarray(run.mu(contexts[b0:b1, -1]))).to(
                    device=base.mu.device, dtype=base.mu.dtype)[:, None]                             # (B, 1, K) mu_v
                static[b0:b1] = _reference_means(last, table)[:, 0].float().cpu().numpy()
            else:
                mu[:, b0:b1] = torch.stack([b.mu[:, -1] for b in layers]).float().cpu().numpy()      # (L+1, B, K)
            var[:, b0:b1] = torch.stack([_covariance_diagonal(run, b.sigma[:, -1])
                                         for b in layers]).float().cpu().numpy()                     # (L+1, B, K)
            for belief in layers[1:]:
                sigma_flag = sigma_flag & _equal_flag(belief.sigma, base.sigma, device)
    sigma_moves = not bool(sigma_flag)
    resplit = split_log.record()
    resplit["skipped_stream_positions"] = [int(pos[i]) for i in split_log.skipped]
    result["forward_resplit"] = resplit
    if split_log.skipped:                                               # drop occurrences that failed alone
        keep = np.ones(m, dtype=bool)
        keep[split_log.skipped] = False
        pos, occ_code, contexts = pos[keep], occ_code[keep], contexts[keep]
        mu, var = mu[:, keep], var[:, keep]
        static = None if static is None else static[keep]
        m = int(keep.sum())
    if m == 0:
        raise RuntimeError("every occurrence window failed the model's exact-symmetry check")
    bounds = np.r_[0, np.flatnonzero(np.diff(occ_code)) + 1, m]
    kept = [words[int(occ_code[a])] for a in bounds[:-1]]
    vectors_by_word = {w: mu[:, a:b] for w, a, b in zip(kept, bounds[:-1], bounds[1:])}
    result.update(
        word_ids=np.asarray(kept, dtype=np.int64),
        vectors=vectors_by_word,
        sigma=({w: var[:, a:b] for w, a, b in zip(kept, bounds[:-1], bounds[1:])} if sigma_moves else None),
        contexts={w: contexts[a:b] for w, a, b in zip(kept, bounds[:-1], bounds[1:])},
        positions={w: pos[a:b] for w, a, b in zip(kept, bounds[:-1], bounds[1:])},
        n_occurrences={w: int(b - a) for w, a, b in zip(kept, bounds[:-1], bounds[1:])},
        n_forward_tokens=int(pos.size + len(split_log.skipped)) * int(context_len), n_layers=L,
        static_reference=(None if static is None else
                          {w: static[a:b].astype(np.float64).mean(axis=0)
                           for w, a, b in zip(kept, bounds[:-1], bounds[1:])}),
        seconds=time.perf_counter() - t_start,
    )
    if chart == "reference":
        notes.append("occurrence vectors are the reference means ξ = U⁻¹μ at the occurrence (the coordinates the "
                     "reference-coordinate decoder reads)")
    if dropped:
        notes.append(f"occurrence budget of {budget_tokens} tokens ({budget_tokens // context_len} windows of "
                     f"{context_len}) exhausted: {len(dropped)} of {len(words)} requested words dropped")
    if missing:
        notes.append(f"{len(missing)} requested words never occur in the {split!r} split with full left context")
    if split == "train":
        notes.append("occurrence contexts come from the TRAIN split (text the model was fit on)")
    return result


def _battery_summary(view: VectorView, ctx: AnalysisContext, rng: np.random.Generator,
                     n_boot: int) -> Dict[str, Any]:
    r"""Analogy accuracy (3CosAdd, ONLY-B), curated answer ranks and field clustering of one view."""
    rec: Dict[str, Any] = {"n_words": view.n}
    notes: List[str] = []

    def analogies() -> Dict[str, Any]:
        bench = vectors.analogy_benchmark(view, ctx.analogies, rng, n_boot=n_boot)
        if bench["n_covered"] == 0:
            return {"status": "skipped", "reason": "no benchmark question has all four words in the view",
                    "n_total": bench["n_total"]}
        summary = {"n_total": bench["n_total"], "n_covered": bench["n_covered"]}
        for group in ("overall", "semantic", "syntactic"):
            methods = bench[group].get("methods", {})
            summary[group] = {"n": bench[group]["n"],
                              **{m: methods[m]["acc"] for m in ("3cosadd", "only_b") if m in methods}}
        return summary

    def curated() -> List[Dict[str, Any]]:
        rows = vectors.analogy_examples(view, ctx.curated_analogies, topk=5)
        return [{"question": (r["a"], r["a_star"], r["b"], r["b_star"]), "category": r["category"],
                 "rank_3cosadd": r["methods"]["3cosadd"]["rank"],
                 "top5_3cosadd": r["methods"]["3cosadd"]["topk"]} for r in rows if r["covered"]]

    def fields() -> Dict[str, Any]:
        res = vectors.field_clustering(view, ctx.lexicon, rng, log_count=ctx.log_count(view.ids),
                                       device=ctx.device)
        if not _ok(res):
            return res
        pur = res["purity"]
        strat = pur.get("frequency_stratified") if isinstance(pur.get("frequency_stratified"), Mapping) else {}
        return {"n_words": res["n_words"], "n_fields": res["n_fields"], "purity": pur["purity"],
                "null_mean": pur["null_mean"], "null_std": pur["null_std"], "z": pur["z"], "p": pur["p"],
                "chance": pur["chance"], "silhouette": res["silhouette"]["silhouette"],
                "stratified_null_mean": strat.get("null_mean"), "stratified_z": strat.get("z")}

    if ctx.analogies:
        _attempt(rec, "analogies", analogies, notes)
    else:
        rec["analogies"] = {"status": "skipped", "reason": "no benchmark analogies in the context"}
    if ctx.curated_analogies:
        _attempt(rec, "curated", curated, notes)
    else:
        rec["curated"] = []
    if ctx.lexicon:
        _attempt(rec, "fields", fields, notes)
    else:
        rec["fields"] = {"status": "skipped", "reason": "no lexicon in the context"}
    rec["notes"] = notes
    return rec


def layerwise_semantics(occ: Dict[str, Any], run: Any, ctx: AnalysisContext) -> Dict[str, Any]:
    r"""The embedding battery on the layerwise static distillation of the occurrence beliefs.

    For layer ``l`` the distilled vector of word ``w`` is the mean of its occurrence means,
    :math:`\bar\mu^{(l)}_w = \frac1{n_w}\sum_c \mu^{(l)}_{w,c}` (Bommasani, Davis & Cardie 2020; the
    mean of the moment-matched pooled Gaussian). Each layer's table, the static prior-mean table
    ``mu_v`` and the decode table (``output_proj_weight`` or the decode means) of the SAME words run
    through :func:`vectors.analogy_benchmark` (Google questions whose four words are all present),
    :func:`vectors.analogy_examples` (curated) and :func:`vectors.field_clustering`. Retrieval is
    among the collected words only, so accuracies compare layers and tables with each other, not
    with the candidate-vocabulary battery of the embeddings section. The occurrence means and the
    static table are those of the occurrences' chart (``occ["chart"]``): under a reference-coordinate
    decoder :math:`\xi = U^{-1}\mu` and :math:`U^{-1}\mu_v` at the occurrence position
    (``occ["static_reference"]``).
    """
    words = np.asarray(occ["word_ids"], dtype=np.int64)
    chart = occ.get("chart", "local")
    out: Dict[str, Any] = {"n_words": int(words.size), "chart": chart, "layers": [], "notes": []}
    if words.size < 4:
        out.update(status="skipped", reason=f"only {words.size} words have occurrences")
        return out
    n_boot = int(_opt(ctx, "n_boot"))
    L = int(occ["n_layers"])
    for l in range(L + 1):
        X = np.stack([occ["vectors"][int(w)][l].astype(np.float64).mean(axis=0) for w in words])   # (w, K)
        view = VectorView(name=f"contextual_layer{l}", label=f"layer {l} (distilled)", ids=words, X=X,
                          group="contextual")
        out["layers"].append({"layer": l, **_battery_summary(view, ctx, _rng(ctx, "semantics"), n_boot)})
    if chart == "reference":
        static = np.stack([occ["static_reference"][int(w)] for w in words])                  # (w, K) U^{-1} mu_v
        static_label = "static prior mean U⁻¹μ_v (reference chart)"
    else:
        static, static_label = run.mu(words), "static prior mean"
    out["static_mu"] = _battery_summary(
        VectorView(name="static_mu", label=static_label, ids=words, X=static, group="mu"),
        ctx, _rng(ctx, "semantics"), n_boot)
    layer0 = np.stack([occ["vectors"][int(w)][0].astype(np.float64).mean(axis=0) for w in words])
    out["layer0_minus_static_max_abs"] = float(np.abs(layer0 - static).max())
    if run.decode_kind == "linear" or run.decode_kind in GAUSSIAN_BANK_DECODE_KINDS:
        table = run.decode_W(words) if run.decode_kind == "linear" else run.decode_mu(words)
        label = "linear decode rows" if run.decode_kind == "linear" else "Gaussian decode means"
        out["decode"] = {"label": label, **_battery_summary(
            VectorView(name="static_decode", label=label, ids=words, X=table, group="decode"),
            ctx, _rng(ctx, "semantics"), n_boot)}
    else:
        out["decode"] = {"status": "skipped", "reason": f"decode kind {run.decode_kind!r}"}
    out["notes"].append("retrieval is among the collected words only; accuracies compare the layers and the "
                        "static tables on the same word set")
    if chart == "reference":
        out["notes"].append("the distilled layers and the static prior mean are in the reference chart ξ = U⁻¹μ, "
                            "the decode table is in the decoder's (reference) coordinates")
    return out


def token_clouds(occ: Dict[str, Any], ctx: AnalysisContext) -> Dict[str, Any]:
    r"""Occurrence clouds of the ambiguous words, per layer (Coenen, Reif et al. 2019).

    For every ambiguous word with at least 8 occurrences: per layer the 2-D PCA of its occurrence
    means (centered over the word's occurrences), the self-similarity (mean pairwise cosine), the MEV
    and the total variance :math:`\mathrm{tr} \mathrm{Cov}`. At the last layer a seeded 2-means split
    of the centered unit rows (exploratory: no sense labels exist offline) with, per cluster, the
    contexts closest to its centroid.
    """
    from sklearn.cluster import KMeans

    rng = _rng(ctx, "clouds")
    out: Dict[str, Any] = {"words": {}, "skipped": {}}
    for w in ctx.ambiguous_ids:
        w = int(w)
        if w not in occ["vectors"]:
            out["skipped"][str(w)] = "no occurrences collected"
            continue
        V = occ["vectors"][w].astype(np.float64)                         # (L+1, n, K)
        n = V.shape[1]
        if n < _CLOUD_MIN_OCCURRENCES:
            out["skipped"][str(w)] = f"{n} occurrences (< {_CLOUD_MIN_OCCURRENCES})"
            continue
        layers = []
        for l in range(V.shape[0]):
            X = V[l]
            Xc = X - X.mean(axis=0, keepdims=True)
            _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
            total = float((s * s).sum())
            Z = stats.unit_rows(X)
            sum_z = Z.sum(axis=0)
            layers.append({
                "pca": Xc @ Vt[:2].T if Vt.shape[0] >= 2 else np.c_[Xc @ Vt[:1].T, np.zeros(n)],
                "explained": (s[:2] ** 2 / total) if total > 0 else np.zeros(2),
                "self_similarity": float((sum_z @ sum_z - n) / (n * (n - 1))),
                "mev": _mev(X), "total_variance": float(Xc.var(axis=0).sum()),
            })
        last = V[-1] - V[-1].mean(axis=0, keepdims=True)
        norms = np.linalg.norm(last, axis=1, keepdims=True)
        Zl = last / np.where(norms > 0, norms, 1.0)
        with warnings.catch_warnings():                                  # sklearn's Windows/MKL thread-count advisory
            warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak")
            km = KMeans(n_clusters=2, n_init=4, random_state=int(rng.integers(0, 2 ** 31 - 1))).fit(Zl)
        clusters = []
        for c in range(2):
            members = np.flatnonzero(km.labels_ == c)
            dist = np.linalg.norm(Zl[members] - km.cluster_centers_[c], axis=1)
            nearest = members[np.argsort(dist, kind="stable")[:4]]
            clusters.append({"size": int(members.size),
                             "examples": [_snippet(ctx.vocab, occ["contexts"][w][i]) for i in nearest]})
        out["words"][str(w)] = {"token": w, "label": ctx.display(w), "n": int(n), "layers": layers,
                                "labels_last": km.labels_.astype(np.int64), "clusters_last": clusters}
    return out


# ----------------------------------------------------------------------------------------------
# Sigma dynamics (only when Sigma moves)
# ----------------------------------------------------------------------------------------------

def _sigma_dynamics(bank: ContextualBank, lens: Optional[Dict[str, Any]], ctx: AnalysisContext) -> Dict[str, Any]:
    r"""Per-layer sharpening :math:`\Delta\log\det = \sum_k \log\Sigma^{(l)}_{kk} - \sum_k\log\Sigma^{(0)}_{kk}`
    (diagonal) and, on the logit-lens tokens, Spearman of the final-layer :math:`\log\det` with the
    predictive entropy and the surprisal."""
    if bank.sigma is None:
        return {"status": "skipped", "reason": "Σ is frozen in context"}
    rng = _rng(ctx, "sigma")
    logdet0 = np.log(bank.sigma0.astype(np.float64)).sum(axis=1)
    per_layer = []
    for l in range(bank.n_layers + 1):
        logdet = np.log(bank.sigma[l].astype(np.float64)).sum(axis=1)
        per_layer.append({"layer": l, **_quantiles(logdet - logdet0),
                          "mean": float((logdet - logdet0).mean())})
    out: Dict[str, Any] = {"delta_logdet": per_layer, "diagonal_only": bank.sigma_kind == "full"}
    if _ok(lens) and "arrays" in lens:
        n = lens["arrays"]["entropy"][-1].size
        rows = np.flatnonzero(np.isin(bank.seq_index, lens["window_index"]))   # window-major, as the lens
        rows = rows[bank.targets[rows] >= 0]
        logdet_L = np.log(bank.sigma[-1, rows].astype(np.float64)).sum(axis=1)
        if logdet_L.size != n:
            raise RuntimeError(f"lens tokens ({n}) and bank rows ({logdet_L.size}) are not aligned")
        if n >= 3:
            # Tokens of one window are dependent: resample whole windows (audit 2026-10-08 gap-stats#7).
            windows = bank.seq_index[rows]
            out["spearman_entropy"] = stats.spearman_ci(logdet_L, lens["arrays"]["entropy"][-1], rng,
                                                        groups=windows)
            out["spearman_surprisal"] = stats.spearman_ci(logdet_L, lens["arrays"]["ce"][-1], rng,
                                                          groups=windows)
    return out


# ----------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------

def _layer_axis(ax: Any, L: int) -> None:
    ax.set_xticks(range(L + 1))
    ax.set_xlim(-0.3, L + 0.3)
    ax.set_xlabel("layer (0 = stack input)")


def _top_legend(ax: Any, *, ncol: int = 2, fontsize: float = 6.0) -> None:
    r"""Legend in a band added above the data: the y-range is extended so that the top
    :math:`p = 0.09 \cdot \mathrm{rows} + 0.04` of the axes is empty, then the legend is placed there."""
    handles, _ = ax.get_legend_handles_labels()
    rows = math.ceil(len(handles) / ncol)
    frac = 0.09 * rows + 0.04
    lo, hi = ax.get_ylim()
    if ax.get_yscale() == "linear":
        ax.set_ylim(lo, (hi - frac * lo) / (1.0 - frac))
    ax.legend(loc="upper center", ncol=ncol, fontsize=fontsize, columnspacing=1.0, handlelength=1.8)


def _ci_line(ax: Any, recs: Sequence[Mapping[str, Any]], *, key_mean: str = "mean", color: str,
             label: str, marker: str = "o", ls: str = "-") -> None:
    x = np.arange(len(recs))
    y = np.array([r[key_mean] for r in recs], dtype=np.float64)
    lo = np.array([r["ci_lo"] for r in recs], dtype=np.float64)
    hi = np.array([r["ci_hi"] for r in recs], dtype=np.float64)
    ax.errorbar(x, y, yerr=[np.clip(y - lo, 0, None), np.clip(hi - y, 0, None)], color=color, marker=marker,
                ms=3.5, lw=1.2, ls=ls, capsize=2, label=label)


def _plot_logit_lens(ctx: AnalysisContext, lens: Dict[str, Any]) -> List[Path]:
    import matplotlib.pyplot as plt

    L = len(lens["layers"]) - 1
    fig, axes = plt.subplots(1, 4, figsize=(10.0, 2.6), constrained_layout=True)
    ax = axes[0]
    _ci_line(ax, lens["ce"], color=OKABE_ITO[0], label="lens CE")
    if lens.get("unigram_ce") is not None:
        ax.axhline(lens["unigram_ce"]["mean"], color="#555555", ls="--", lw=1.0, label="unigram readout")
    ax.set_ylabel("next-token CE (nats)")
    ax.legend(loc="best")
    ax = axes[1]
    _ci_line(ax, lens["acc"], color=OKABE_ITO[0], label="top-1 accuracy")
    _ci_line(ax, lens["agree"], color=OKABE_ITO[1], label="top-1 = final layer", marker="s")
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("fraction of tokens")
    ax.legend(loc="best")
    ax = axes[2]
    _ci_line(ax, lens["kl"], color=OKABE_ITO[2], label=r"KL$(p_L \| p_l)$")
    _ci_line(ax, lens["entropy"], color=OKABE_ITO[3], label=r"entropy $H(p_l)$", marker="s")
    ax.set_ylabel("nats")
    ax.legend(loc="best")
    for a in axes[:3]:
        _layer_axis(a, L)
    ax = axes[3]
    ce_pos = np.asarray(lens["ce_by_position"], dtype=np.float64)       # (L+1, N)
    N = ce_pos.shape[1]
    edges = np.unique(np.round(np.geomspace(1, N + 1, _POSITION_BINS + 1)).astype(int)) - 1
    edges[-1] = N
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)            # all-NaN bins (no scored token)
        binned = np.stack([np.nanmean(ce_pos[:, a:b], axis=1) for a, b in zip(edges[:-1], edges[1:])], axis=1)
    lo, hi = np.nanpercentile(binned, [2.0, 98.0])                    # robust scale: position 0 is extreme
    im = ax.imshow(binned, aspect="auto", origin="lower", cmap="viridis", vmin=lo, vmax=hi)
    ax.set_yticks(range(L + 1))
    ax.set_ylabel("layer")
    ticks = np.linspace(0, binned.shape[1] - 1, min(5, binned.shape[1])).round().astype(int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(edges[t]) for t in ticks])
    ax.set_xlabel("position (log bins, start)")
    fig.colorbar(im, ax=ax, label="CE (nats)", shrink=0.9, extend="both")
    for a, letter in zip(axes, "abcd"):
        panel_label(a, letter)
    fig.suptitle(f"Logit lens: every layer read out by the model's final decode "
                 f"({lens['n_tokens']} tokens, {lens['n_windows']} windows; 95% window-bootstrap CIs)",
                 fontsize=9)
    return save_figure(fig, ctx, _SECTION, "logit_lens")


def _plot_geometry(ctx: AnalysisContext, geo: Dict[str, Any]) -> List[Path]:
    import matplotlib.pyplot as plt

    layers = geo["layers"]
    L = len(layers) - 1
    x = np.arange(L + 1)

    def series(key: str, sub: Optional[str] = None) -> np.ndarray:
        vals = []
        for rec in layers:
            v = rec.get(key)
            if isinstance(v, dict):
                v = v.get(sub) if sub is not None and v.get("status") != "failed" else None
            vals.append(np.nan if v is None or isinstance(v, dict) else float(v))
        return np.asarray(vals, dtype=np.float64)

    fig, axes = plt.subplots(1, 4, figsize=(10.0, 2.5), constrained_layout=True)
    axes[0].plot(x, series("anisotropy_raw"), "o-", color=OKABE_ITO[0], ms=3.5, label="raw")
    axes[0].plot(x, series("anisotropy_centered"), "s--", color=OKABE_ITO[1], ms=3.5, label="centered")
    axes[0].set_ylabel("mean cosine of random pairs")
    axes[0].legend()
    axes[1].plot(x, series("effective_rank"), "o-", color=OKABE_ITO[0], ms=3.5, label="effective rank")
    axes[1].plot(x, series("participation_ratio"), "s--", color=OKABE_ITO[1], ms=3.5, label="participation ratio")
    axes[1].axhline(geo["K"], color="#777777", lw=0.8, ls=":", label=f"K = {geo['K']}")
    axes[1].set_ylabel("dimensions")
    axes[1].legend()
    axes[2].plot(x, series("isoscore"), "o-", color=OKABE_ITO[2], ms=3.5)
    axes[2].set_ylim(bottom=0)
    axes[2].set_ylabel("IsoScore")
    axes[3].plot(x, series("twonn", "mle"), "o-", color=OKABE_ITO[3], ms=3.5, label="TwoNN MLE")
    axes[3].plot(x, series("twonn", "fit"), "s--", color=OKABE_ITO[4], ms=3.5, label="TwoNN fit")
    axes[3].set_ylabel("intrinsic dimension")
    axes[3].legend()
    for a, letter in zip(axes, "abcd"):
        _layer_axis(a, L)
        panel_label(a, letter)
    fig.suptitle(f"Geometry of the contextual means per layer (n = {geo['n']} positions)", fontsize=9)
    return save_figure(fig, ctx, _SECTION, "geometry")


def _plot_cka(ctx: AnalysisContext, cka: Dict[str, Any]) -> List[Path]:
    import matplotlib.pyplot as plt

    M = np.asarray(cka["debiased"], dtype=np.float64)
    L = M.shape[0] - 1
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.0, 1.25]})
    ax = axes[0]
    im = ax.imshow(M, vmin=min(0.0, float(np.nanmin(M))), vmax=1.0, cmap="viridis", origin="lower")
    ax.set_xticks(range(L + 1))
    ax.set_yticks(range(L + 1))
    ax.set_xlabel("layer")
    ax.set_ylabel("layer")
    if L + 1 <= 10:
        for i in range(L + 1):
            for j in range(L + 1):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if M[i, j] < 0.6 else "black")
    fig.colorbar(im, ax=ax, label="debiased linear CKA", shrink=0.85)
    ax = axes[1]
    x = np.arange(L + 1)
    ax.plot(x, cka["prior_mu"]["debiased"], "o-", color=OKABE_ITO[0], ms=3.5, label=r"token prior $\mu_{v_i}$")
    ax.plot(x, cka["prior_mu"]["biased"], "o:", color=OKABE_ITO[0], ms=2.5, lw=0.8, label="(biased)")
    nxt = cka.get("next_token_decode")
    if nxt is not None:
        ax.plot(x, nxt["debiased"], "s-", color=OKABE_ITO[1], ms=3.5, label="next-token decode row")
        ax.plot(x, nxt["biased"], "s:", color=OKABE_ITO[1], ms=2.5, lw=0.8, label="(biased)")
    ax.set_ylabel("linear CKA with layer")
    _layer_axis(ax, L)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    for a, letter in zip(axes, "ab"):
        panel_label(a, letter)
    fig.suptitle(f"Representational similarity of the contextual means (n = {cka['n']} positions)", fontsize=9)
    return save_figure(fig, ctx, _SECTION, "cka")


def _plot_contextuality(ctx: AnalysisContext, cx: Dict[str, Any]) -> List[Path]:
    import matplotlib.pyplot as plt

    layers = cx["layers"]
    L = len(layers) - 1
    x = np.arange(L + 1)
    fig, axes = plt.subplots(1, 4, figsize=(10.4, 2.6), constrained_layout=True)
    ax = axes[0]
    _ci_line(ax, [r["self_similarity_adjusted"] for r in layers], color=OKABE_ITO[0], label="all types")
    _ci_line(ax, [r["intra_similarity_adjusted"] for r in layers], color=OKABE_ITO[7], label="intra-window",
             marker="s", ls="--")
    for i, name in enumerate(("function word", "content word", "capitalized word", "punctuation")):
        ys = [r["by_class"].get(name, {}).get("self_similarity_adjusted", np.nan) for r in layers]
        if np.isfinite(ys).any():
            ax.plot(x, ys, ".-", lw=0.8, ms=3, color=OKABE_ITO[1 + i], label=name)
    ax.axhline(0.0, color="#999999", lw=0.6)
    ax.set_ylabel("similarity − random baseline")
    _top_legend(ax)
    ax = axes[1]
    _ci_line(ax, [r["mev_adjusted"] for r in layers], color=OKABE_ITO[2], label="MEV − baseline")
    ax.axhline(0.0, color="#999999", lw=0.6)
    ax.set_ylabel("max. explainable variance")
    ax.legend()
    ax = axes[2]
    kls = [r["kl_to_stack_input"] for r in layers]
    med = np.array([k["median"] for k in kls])
    ax.fill_between(x, [k["q25"] for k in kls], [k["q75"] for k in kls], color=OKABE_ITO[3], alpha=0.25,
                    lw=0, label="IQR")
    ax.plot(x, med, "o-", color=OKABE_ITO[3], ms=3.5, label="median")
    ax.plot(x, [k["mean"] for k in kls], "s--", color=OKABE_ITO[3], ms=3, lw=0.8, label="mean")
    ax.set_yscale("symlog", linthresh=1e-2)
    ax.set_ylabel(r"KL$(q^{(l)}_i \| q^{(0)}_i)$ (nats)")
    ax.legend()
    ax = axes[3]
    _ci_line(ax, [r["variance_share"] for r in layers], color=OKABE_ITO[4], label=r"$\rho$ (trace ratio)")
    _ci_line(ax, [r["variance_share_per_coordinate"] for r in layers], color=OKABE_ITO[5],
             label=r"$\rho$ (per coordinate)", marker="s", ls="--")
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("within-type variance share")
    ax.legend()
    for a, letter in zip(axes, "abcd"):
        _layer_axis(a, L)
        panel_label(a, letter)
    fig.suptitle(f"Contextuality ({cx['n_types']} types with ≥ {cx['min_occurrences']} occurrences; "
                 f"{cx['kl_form']})", fontsize=9)
    return save_figure(fig, ctx, _SECTION, "contextuality")


def _series(sem: Dict[str, Any], getter: Callable[[Dict[str, Any]], Any]) -> np.ndarray:
    vals = []
    for rec in sem["layers"]:
        try:
            v = getter(rec)
        except (KeyError, TypeError):
            v = None
        vals.append(np.nan if v is None else float(v))
    return np.asarray(vals, dtype=np.float64)


def _ref(rec: Dict[str, Any], getter: Callable[[Dict[str, Any]], Any]) -> Optional[float]:
    try:
        v = getter(rec)
    except (KeyError, TypeError):
        return None
    return None if v is None else float(v)


def _plot_layerwise_semantics(ctx: AnalysisContext, sem: Dict[str, Any]) -> List[Path]:
    import matplotlib.pyplot as plt

    L = len(sem["layers"]) - 1
    x = np.arange(L + 1)
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.6), constrained_layout=True)
    panels = (
        (axes[0], lambda r: r["analogies"]["overall"]["3cosadd"]["mean"], "3CosAdd accuracy",
         lambda r: r["analogies"]["overall"]["only_b"]["mean"]),
        (axes[1], lambda r: r["fields"]["purity"], "field kNN purity@10", lambda r: r["fields"]["null_mean"]),
        (axes[2], lambda r: r["fields"]["silhouette"], "field silhouette (cosine)", None),
    )
    for ax, get, ylabel, baseline in panels:
        ax.plot(x, _series(sem, get), "o-", color=OKABE_ITO[0], ms=3.5, label="distilled layer")
        if baseline is not None:
            label = "ONLY-B" if ylabel.startswith("3Cos") else "permutation null"
            ax.plot(x, _series(sem, baseline), "s:", color="#777777", ms=3, lw=0.8, label=label)
        for key, color, ls in (("static_mu", OKABE_ITO[1], "--"), ("decode", OKABE_ITO[2], "-.")):
            ref = _ref(sem.get(key, {}), get)
            if ref is not None:
                name = "static prior μ" if key == "static_mu" else sem["decode"].get("label", "decode")
                ax.axhline(ref, color=color, ls=ls, lw=1.0, label=name)
        ax.set_ylabel(ylabel)
        _layer_axis(ax, L)
    _top_legend(axes[0])
    _top_legend(axes[1])
    for a, letter in zip(axes, "abc"):
        panel_label(a, letter)
    n_cov = _ref(sem["layers"][0], lambda r: r["analogies"]["n_covered"])
    fig.suptitle(f"Layerwise static distillation: mean contextual μ over training contexts "
                 f"({sem['n_words']} words; {int(n_cov) if n_cov is not None else 0} analogy questions covered)",
                 fontsize=9)
    return save_figure(fig, ctx, _SECTION, "layerwise_semantics")


def _plot_token_clouds(ctx: AnalysisContext, clouds: Dict[str, Any]) -> List[Path]:
    import matplotlib.pyplot as plt

    words = sorted(clouds["words"].values(), key=lambda w: -w["n"])[:4]
    L = len(words[0]["layers"]) - 1
    cols = list(range(L + 1)) if L + 1 <= 7 else sorted(set(np.linspace(0, L, 7).round().astype(int).tolist()))
    fig, axes = plt.subplots(len(words), len(cols), figsize=(1.45 * len(cols) + 0.4, 1.45 * len(words) + 0.5),
                             constrained_layout=True, squeeze=False)
    for i, w in enumerate(words):
        colors = np.array([OKABE_ITO[0], OKABE_ITO[1]])[w["labels_last"]]
        for j, l in enumerate(cols):
            ax = axes[i, j]
            xy = np.asarray(w["layers"][l]["pca"], dtype=np.float64)
            ax.scatter(xy[:, 0], xy[:, 1], s=6, c=colors, lw=0)
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ("top", "right"):
                ax.spines[side].set_visible(True)
            if i == 0:
                ax.set_title(f"layer {l}", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{w['label']} (n={w['n']})".replace("$", r"\$"), fontsize=8)
            ax.text(0.02, 0.02, f"self-sim {w['layers'][l]['self_similarity']:.2f}", transform=ax.transAxes,
                    fontsize=5.5, color="#333333")
    fig.suptitle("Occurrence clouds of ambiguous words (2-D PCA per layer; colors: exploratory 2-means split "
                 "at the last layer)", fontsize=8)
    return save_figure(fig, ctx, _SECTION, "token_clouds")


def _plot_sigma(ctx: AnalysisContext, sig: Dict[str, Any]) -> List[Path]:
    import matplotlib.pyplot as plt

    per = sig["delta_logdet"]
    L = len(per) - 1
    x = np.arange(L + 1)
    fig, ax = plt.subplots(1, 1, figsize=(3.6, 2.6), constrained_layout=True)
    ax.fill_between(x, [p["q25"] for p in per], [p["q75"] for p in per], color=OKABE_ITO[0], alpha=0.25, lw=0,
                    label="IQR")
    ax.plot(x, [p["median"] for p in per], "o-", color=OKABE_ITO[0], ms=3.5, label="median")
    ax.axhline(0.0, color="#999999", lw=0.6)
    ax.set_ylabel(r"$\log\det\Sigma^{(l)} - \log\det\Sigma^{(0)}$" + (" (diag.)" if sig["diagonal_only"] else ""))
    _layer_axis(ax, L)
    ax.legend()
    title = "Contextual covariance sharpening"
    if "spearman_entropy" in sig:
        title += (f"\nSpearman of final-layer log det Σ with predictive entropy "
                  f"{sig['spearman_entropy']['rho']:.2f}, with surprisal {sig['spearman_surprisal']['rho']:.2f}")
    ax.set_title(title, fontsize=8)
    return save_figure(fig, ctx, _SECTION, "sigma")


# ----------------------------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------------------------

def _lens_table(ctx: AnalysisContext, lens: Dict[str, Any]) -> str:
    rows = []
    for l, name in enumerate(lens["layer_names"]):
        rows.append({"layer": f"{l} ({name})", "CE": lens["ce"][l]["mean"],
                     "CE 95% CI": f"[{lens['ce'][l]['ci_lo']:.3f}, {lens['ce'][l]['ci_hi']:.3f}]",
                     "top-1 acc": lens["acc"][l]["mean"], "agree final": lens["agree"][l]["mean"],
                     "KL(p_L‖p_l)": lens["kl"][l]["mean"], "entropy": lens["entropy"][l]["mean"]})
    text = "Logit lens per layer (means over tokens; CIs by window bootstrap)\n\n" + markdown_table(rows)
    ex = lens.get("example")
    if ex is not None:
        P = ex["window"].size
        cols = ["row"] + [f"p{p}" for p in range(P)]
        erows = [{"row": "input", **{f"p{p}": ctx.display(int(ex["window"][p])) for p in range(P)}},
                 {"row": "target", **{f"p{p}": ctx.display(int(ex["targets"][p])) for p in range(P)}}]
        for l in range(ex["top1"].shape[0]):
            erows.append({"row": f"top-1 @ layer {l}",
                          **{f"p{p}": ctx.display(int(ex["top1"][l, p])) for p in range(P)}})
        text += "\nTop-1 readout of the first lens window, per layer\n\n" + markdown_table(erows, cols)
    return text


def _semantics_table(ctx: AnalysisContext, sem: Dict[str, Any]) -> str:
    def row(name: str, rec: Dict[str, Any]) -> Dict[str, Any]:
        an = rec.get("analogies", {})
        fl = rec.get("fields", {})
        ok_an, ok_fl = _ok(an), _ok(fl)
        return {"table": name,
                "questions": an.get("n_covered") if ok_an else None,
                "3CosAdd": an["overall"]["3cosadd"]["mean"] if ok_an else None,
                "ONLY-B": an["overall"]["only_b"]["mean"] if ok_an else None,
                "field purity": fl.get("purity") if ok_fl else None,
                "purity z": fl.get("z") if ok_fl else None,
                "silhouette": fl.get("silhouette") if ok_fl else None}

    rows = [row(f"layer {r['layer']}", r) for r in sem["layers"]]
    rows.append(row("static prior μ", sem["static_mu"]))
    if _ok(sem.get("decode")):
        rows.append(row(sem["decode"]["label"], sem["decode"]))
    text = (f"Layerwise static distillation ({sem['n_words']} words; retrieval among these words)\n\n"
            + markdown_table(rows))
    tables = [(f"layer {r['layer']}", r) for r in sem["layers"]] + [("static μ", sem["static_mu"])]
    if _ok(sem.get("decode")):
        tables.append(("decode", sem["decode"]))
    last = f"layer {sem['layers'][-1]['layer']}"
    ranks: Dict[Tuple[int, ...], Dict[str, Any]] = {}
    for name, rec in tables:
        for q in (rec["curated"] if isinstance(rec.get("curated"), list) else []):
            key = tuple(int(t) for t in q["question"])
            row_q = ranks.setdefault(key, {"analogy": f"{ctx.display(key[0])}:{ctx.display(key[1])}::"
                                                      f"{ctx.display(key[2])}:{ctx.display(key[3])}"})
            row_q[name] = q["rank_3cosadd"]
            if name == last:
                row_q["top-1 (last layer)"] = ctx.display(q["top5_3cosadd"][0])
    if ranks:
        cols = ["analogy"] + [name for name, _ in tables] + ["top-1 (last layer)"]
        text += ("\nCurated analogies: rank of the answer under 3CosAdd (queries excluded), per table\n\n"
                 + markdown_table(list(ranks.values()), cols))
    return text


def _clouds_table(clouds: Dict[str, Any]) -> str:
    rows = []
    for w in clouds["words"].values():
        for c, cl in enumerate(w["clusters_last"]):
            rows.append({"word": w["label"], "cluster": c, "size": cl["size"],
                         "contexts nearest the centroid": " ‖ ".join(cl["examples"])})
    return ("Exploratory 2-means split of the last-layer occurrence means (no sense labels)\n\n"
            + markdown_table(rows))


# ----------------------------------------------------------------------------------------------
# Section driver
# ----------------------------------------------------------------------------------------------

def _interleave(a: Sequence[int], b: Sequence[int]) -> List[int]:
    out: List[int] = []
    for i in range(max(len(a), len(b))):
        if i < len(a):
            out.append(int(a[i]))
        if i < len(b):
            out.append(int(b[i]))
    return out


def occurrence_words(ctx: AnalysisContext) -> Tuple[List[int], Dict[int, int]]:
    r"""Priority-ordered words for the occurrence budget and their per-word overrides.

    Order: the ambiguous words (``cloud_per_word`` occurrences each), the curated-analogy words, then
    lexicon words (round-robin over fields, most frequent first) interleaved with benchmark-analogy
    words (most questions first), so that a budget cut keeps both fields and analogies represented.
    """
    counts = ctx.counts

    def by_count(ids: Sequence[int]) -> List[int]:
        ids = [int(i) for i in ids]
        return sorted(ids, key=lambda i: (-int(counts[i]), i)) if counts is not None else ids

    fields = [by_count(ids) for _, ids in sorted(ctx.lexicon.items())]
    lexicon_rr: List[int] = []
    for i in range(max((len(f) for f in fields), default=0)):
        lexicon_rr.extend(f[i] for f in fields if i < len(f))
    participation: Dict[int, int] = {}
    for q in ctx.analogies:
        for t in (q.a, q.a_star, q.b, q.b_star):
            participation[int(t)] = participation.get(int(t), 0) + 1
    google = sorted(participation, key=lambda t: (-participation[t], -(int(counts[t]) if counts is not None else 0), t))
    curated = [int(t) for q in ctx.curated_analogies for t in (q.a, q.a_star, q.b, q.b_star)]
    ambiguous = [int(t) for t in ctx.ambiguous_ids]
    words = list(dict.fromkeys(ambiguous + curated + _interleave(lexicon_rr, google)))
    overrides = {t: int(_opt(ctx, "cloud_per_word")) for t in ambiguous}
    return words, overrides


def _metrics_view(lens: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in lens.items() if k not in ("arrays", "clusters", "window_index")}


def _occurrence_metrics(occ: Dict[str, Any]) -> Dict[str, Any]:
    keep = ("split", "context_len", "per_word", "budget_tokens", "missing", "dropped_budget", "partial",
            "n_forward_tokens", "forward_resplit", "chart", "seconds", "notes")
    out = {k: occ[k] for k in keep if k in occ}
    out["n_words"] = int(np.asarray(occ.get("word_ids", [])).size)
    out["n_occurrences_total"] = int(sum(occ.get("n_occurrences", {}).values()))
    return out


def _clouds_metrics(clouds: Dict[str, Any]) -> Dict[str, Any]:
    words = {}
    for key, w in clouds["words"].items():
        words[key] = {"label": w["label"], "n": w["n"], "clusters_last": w["clusters_last"],
                      "layers": [{k: v for k, v in rec.items() if k != "pca"} for rec in w["layers"]]}
    return {"words": words, "skipped": clouds["skipped"]}


def _headline(res: SectionResult, lens: Any, cka: Any, cx: Any, sem: Any, bank: ContextualBank) -> None:
    h = res.headline
    h["contextual: Σ / φ move in context (measured)"] = (
        f"{'no' if bank.sigma_frozen else 'yes'} / {'yes' if bank.phi_moves else 'no'}")
    if _ok(lens):
        h["contextual: lens CE layer 0 → final (nats)"] = (
            f"{lens['ce'][0]['mean']:.3f} → {lens['ce'][-1]['mean']:.3f}"
            + (f" (unigram {lens['unigram_ce']['mean']:.3f})" if lens.get("unigram_ce") else ""))
        h["contextual: first layer with top-1 agreement ≥ 0.5 with the final layer"] = lens["first_layer_agree_half"]
        h["contextual: lens/forward max |Δlogit| at the final layer"] = (
            lens["consistency"]["max_abs_logit_diff_final_vs_forward"])
    if _ok(cka):
        h["contextual: debiased CKA(final layer, prior μ)"] = cka["prior_mu"]["debiased"][-1]
        if cka.get("next_token_decode"):
            h["contextual: debiased CKA(final layer, next-token decode row)"] = cka["next_token_decode"]["debiased"][-1]
    if _ok(cx):
        last = cx["layers"][-1]
        h["contextual: final-layer self-similarity − baseline"] = last["self_similarity_adjusted"]["mean"]
        h["contextual: final-layer median KL(q_L ‖ q_0)"] = last["kl_to_stack_input"]["median"]
    if _ok(sem):
        # The final layer, not the best one: a maximum over layers would be selected on the same data.
        final, static = sem["layers"][-1], sem["static_mu"]
        fmt = lambda v: "–" if v is None else f"{v:.3f}"                 # noqa: E731
        acc = {(name, m): _ref(rec, lambda r: r["analogies"]["overall"][m]["mean"])     # called at once
               for name, rec in (("final", final), ("static", static)) for m in ("3cosadd", "only_b")}
        if acc[("final", "3cosadd")] is not None or acc[("static", "3cosadd")] is not None:
            n_cov = _ref(final, lambda r: r["analogies"]["n_covered"])
            n_tot = _ref(final, lambda r: r["analogies"]["n_total"])
            h["contextual: distilled 3CosAdd (ONLY-B) final layer / static μ, retrieval among the collected words"] = (
                f"{fmt(acc[('final', '3cosadd')])} ({fmt(acc[('final', 'only_b')])}) / "
                f"{fmt(acc[('static', '3cosadd')])} ({fmt(acc[('static', 'only_b')])}); layer {final['layer']}, "
                f"retrieval among {sem['n_words']} words, "
                f"{'–' if n_cov is None else int(n_cov)} of {'–' if n_tot is None else int(n_tot)} questions covered")
        pur_L = _ref(final, lambda r: r["fields"]["purity"])
        pur_s = _ref(static, lambda r: r["fields"]["purity"])
        if pur_L is not None or pur_s is not None:
            null = _ref(final, lambda r: r["fields"]["null_mean"])
            n_fw = _ref(final, lambda r: r["fields"]["n_words"])
            h["contextual: distilled field purity final layer / static μ"] = (
                f"{fmt(pur_L)} / {fmt(pur_s)}; layer {final['layer']}, permutation null {fmt(null)}, "
                f"{'–' if n_fw is None else int(n_fw)} lexicon words")


def _pick(obj: Any, *keys: Any) -> Any:
    r"""``obj[k0][k1]...``, or ``None`` when a key is missing or a step on the way failed or was skipped."""
    for key in keys:
        if isinstance(obj, Mapping):
            if obj.get("status") in ("failed", "skipped"):
                return None
            obj = obj.get(key)
        elif isinstance(obj, (list, tuple)) and isinstance(key, int) and -len(obj) <= key < len(obj):
            obj = obj[key]
        else:
            return None
    if isinstance(obj, Mapping) and obj.get("status") in ("failed", "skipped"):
        return None
    return obj


def _nats(value: float) -> str:
    return f"{value:.1f}" if abs(value) >= 10.0 else f"{value:.3f}"


def contextual_summary(metrics: Mapping[str, Any]) -> str:
    r"""Three to six sentences of academic prose with the section's key numbers, for the report.

    Reads only the section ``metrics`` (as returned, or after a strict-JSON round trip): the logit-lens
    cross-entropy at the stack input and the final layer against the unigram readout, the first layer
    whose top-1 prediction agrees with the final layer on at least half of the tokens, the debiased
    CKA of the final layer with the prior mean and the next token's decode row, the trend of the
    baseline-adjusted self-similarity, the median :math:`\mathrm{KL}(q^{(L)}_i \| q^{(0)}_i)`, the
    final-layer distilled 3CosAdd accuracy against ONLY-B with the retrieval pool, and the token
    clouds. A sentence whose inputs are missing, skipped or failed is left out; ``""`` when none
    is available.
    """
    sentences: List[str] = []
    ce = _pick(metrics, "logit_lens", "ce")
    if ce and ce[0].get("mean") is not None and ce[-1].get("mean") is not None:
        L = len(ce) - 1
        split, selection = _pick(metrics, "bank", "split"), _pick(metrics, "bank", "selection_split")
        if split is None:
            tokens = "tokens"
        elif selection is None:
            tokens = f"{split}-split tokens"
        elif split == selection:                        # audit 2026-10-16 F121: not held out from selection
            tokens = f"{split}-split tokens, the split that selected the checkpoint"
        else:
            tokens = f"held-out {split}-split tokens"
        text = (f"Read out through the model's own final decode (logit lens on "
                f"{_pick(metrics, 'logit_lens', 'n_tokens')} {tokens}), the next-token cross-entropy "
                f"{'falls' if ce[-1]['mean'] < ce[0]['mean'] else 'rises'} from {ce[0]['mean']:.3f} nats at the "
                f"stack input to {ce[-1]['mean']:.3f} nats at block {L}")
        unigram = _pick(metrics, "logit_lens", "unigram_ce", "mean")
        text += f", against {unigram:.3f} nats for the context-free unigram readout" if unigram is not None else ""
        first = _pick(metrics, "logit_lens", "first_layer_agree_half")
        if first is not None and int(first) < L:
            text += (f"; the top-1 prediction first agrees with the final layer's on at least half of the "
                     f"tokens at layer {int(first)}")
        elif first is not None:
            text += "; no earlier layer's top-1 prediction agrees with the final layer's on half of the tokens"
        sentences.append(text + ".")
    cka_prior = _pick(metrics, "cka", "prior_mu", "debiased", -1)
    cka_next = _pick(metrics, "cka", "next_token_decode", "debiased", -1)
    if cka_prior is not None:
        text = (f"At the final layer the debiased linear CKA of the contextual means is {cka_prior:.3f} with the "
                "static prior mean of the token")
        text += f" and {cka_next:.3f} with the decode row of the next token" if cka_next is not None else ""
        sentences.append(text + ".")
    layers = _pick(metrics, "contextuality", "layers")
    if layers:
        adj = [_pick(rec, "self_similarity_adjusted", "mean") for rec in layers]
        if all(a is not None for a in adj):
            lo = int(np.argmin(adj))
            verb = "falls" if adj[-1] < adj[0] else "rises"
            text = (f"Self-similarity above the random-position baseline (Ethayarajh 2019; "
                    f"{_pick(metrics, 'contextuality', 'n_types')} token types) {verb} from {adj[0]:.3f} at the "
                    f"stack input to {adj[-1]:.3f} at the final layer")
            if 0 < lo < len(adj) - 1:
                text += f", with its minimum {adj[lo]:.3f} at layer {lo}"
            kl = _pick(layers, -1, "kl_to_stack_input", "median")
            if kl is not None:
                frozen = _pick(metrics, "bank", "sigma_frozen_measured")
                text += (f", and the median displacement KL(q_L ‖ q_0) from the stack input is {_nats(kl)} nats"
                         + (" (with Σ frozen, the Mahalanobis displacement ½ Δμᵀ Σ₀⁻¹ Δμ)" if frozen else ""))
            sentences.append(text + ".")
    sem = _pick(metrics, "layerwise_semantics")
    if sem is not None and sem.get("layers"):
        final, static = sem["layers"][-1], sem.get("static_mu")
        acc = _pick(final, "analogies", "overall", "3cosadd", "mean")
        only_b = _pick(final, "analogies", "overall", "only_b", "mean")
        if acc is not None:
            text = (f"Distilled static vectors (the mean contextual μ of a word over its training contexts) answer "
                    f"{acc:.3f} of the {_pick(final, 'analogies', 'n_covered')} covered Google analogies with "
                    f"3CosAdd at the final layer")
            text += f", against {only_b:.3f} for the ONLY-B baseline on the same pool" if only_b is not None else ""
            static_acc = _pick(static, "analogies", "overall", "3cosadd", "mean")
            text += f" and {static_acc:.3f} for the static prior mean" if static_acc is not None else ""
            sentences.append(text + f"; retrieval is among the {sem.get('n_words')} collected words only, so these "
                             "accuracies are not comparable with the candidate-vocabulary battery of the embeddings "
                             "section.")
    clouds = _pick(metrics, "token_clouds")
    if clouds is not None and "words" in clouds:
        words = list(clouds["words"].values())
        if words:
            self_sim = [w["layers"][-1]["self_similarity"] for w in words]
            names = ", ".join(str(w["label"]) for w in words[:4]) + (", …" if len(words) > 4 else "")
            spread = (f"has final-layer self-similarity {self_sim[0]:.3f}" if len(words) == 1 else
                      f"have final-layer self-similarity between {min(self_sim):.3f} and {max(self_sim):.3f}")
            sentences.append(
                f"Occurrence clouds of {len(words)} ambiguous word{'s' if len(words) != 1 else ''} ({names}) "
                f"{spread}; the two-cluster split of the last layer is exploratory, since no sense labels exist "
                "offline.")
        else:
            sentences.append("No ambiguous word had enough collected occurrences for a token cloud.")
    return " ".join(sentences[:6])


def run_contextual_section(run: Any, ctx: AnalysisContext) -> SectionResult:
    r"""The ``contextual`` section: per-layer bank, logit lens, geometry, CKA, contextuality,
    layerwise static distillation and token clouds; figures under ``figures/contextual``, tables
    under ``tables/contextual``. Budgets come from ``ctx.options`` (keys of
    :data:`CONTEXTUAL_DEFAULTS`). The model is rebuilt with ``run.build_model(ctx.device)`` (float32,
    autocast off) and torch uses ``ctx.threads`` threads for the duration of the section. Every caught
    sub-step failure (at any depth of the metrics, and figures/tables) is listed in
    ``metrics["failed_steps"]`` as ``{"step", "error"}``; the re-split and skipped-window counts of
    the forwards are in ``metrics["forward_resplits"]``. :func:`contextual_summary` turns the metrics
    into prose.
    """
    t_start = time.perf_counter()
    res = SectionResult(name=_SECTION)
    if run.dataset is None:
        res.status = "skipped"
        res.notes.append("the run's dataset is unknown, so no token stream can be read for forward passes")
        return res
    if int(run.n_layers) < 1:
        res.status = "skipped"
        res.notes.append("the model has no blocks: there are no contextual layers")
        return res
    apply_style()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(max(1, int(ctx.threads)))
    timings: Dict[str, float] = {}
    opts = _resolved_options(ctx, run)                                  # device-resolved budgets, recorded
    metrics: Dict[str, Any] = {"options": opts, "device": str(ctx.device), "timings": timings}
    notes = res.notes
    model: Any = None
    try:
        model = run.build_model(ctx.device)
        if run.cfg.amp_dtype is not None:
            notes.append(f"the model was trained with amp_dtype={run.cfg.amp_dtype!r} and is replayed here in "
                         "float32 with autocast disabled")
        t = time.perf_counter()
        split = str(opts["contextual_split"])
        try:
            bank = collect_bank(run, model, ctx, split=split, n_tokens=int(opts["contextual_tokens"]),
                                seq_len=opts["contextual_seq_len"],
                                batch_size=int(opts["contextual_batch_size"]))
        except FileNotFoundError as exc:
            res.status = "skipped"
            notes.append(f"no cached {split!r} token stream for {run.dataset!r}: {exc}")
            return res
        timings["bank"] = time.perf_counter() - t
        notes.extend(bank.notes)
        metrics["bank"] = bank.diagnostics
        metrics["layer_names"] = _layer_names(bank.n_layers, run.cfg.encode_mode)

        t = time.perf_counter()
        lens = _attempt(metrics, "logit_lens", lambda: logit_lens(
            run, model, bank, ctx, max_tokens=int(opts["lens_tokens"]),
            chunk_tokens=int(opts["lens_chunk_tokens"])), notes)
        timings["logit_lens"] = time.perf_counter() - t
        if _ok(lens):
            notes.extend(lens["notes"])
            metrics["logit_lens"] = _metrics_view(lens)
        for key, fn in (("geometry", lambda: layer_geometry(bank, ctx)),
                        ("cka", lambda: layer_cka(bank, run, ctx)),
                        ("contextuality", lambda: contextuality(bank, run, ctx)),
                        ("sigma", lambda: _sigma_dynamics(bank, lens if _ok(lens) else None, ctx))):
            t = time.perf_counter()
            _attempt(metrics, key, fn, notes)
            timings[key] = time.perf_counter() - t
        if _ok(metrics["contextuality"]):
            notes.extend(metrics["contextuality"]["notes"])

        occ: Any = None
        words, overrides = occurrence_words(ctx)
        if not words:
            metrics["occurrences"] = {"status": "skipped", "reason": "no resolved words in the context"}
            notes.append("no resolved lexicon/analogy/ambiguous words: occurrence analyses skipped")
        else:
            t = time.perf_counter()
            occ_split = str(opts["occurrence_split"])
            try:
                occ = collect_occurrences(
                    run, model, ctx, words, split=occ_split, per_word=int(opts["occurrence_per_word"]),
                    context_len=int(opts["context_len"]), budget_tokens=int(opts["occurrence_tokens"]),
                    per_word_overrides=overrides, batch_size=int(opts["occurrence_batch_size"]))
            except FileNotFoundError as exc:
                metrics["occurrences"] = {"status": "skipped", "reason": str(exc)}
                notes.append(f"no cached {occ_split!r} stream: occurrence analyses skipped ({exc})")
                occ = None
            except Exception as exc:  # recorded, the rest of the section survives
                metrics["occurrences"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                                          "traceback": traceback.format_exc()[-3000:]}
                notes.append(f"contextual occurrences failed: {type(exc).__name__}: {exc}")
                occ = None
            else:
                metrics["occurrences"] = _occurrence_metrics(occ)
                notes.extend(occ["notes"])
            timings["occurrences"] = time.perf_counter() - t
        sem: Any = None
        clouds: Any = None
        if occ is not None:
            t = time.perf_counter()
            sem = _attempt(metrics, "layerwise_semantics", lambda: layerwise_semantics(occ, run, ctx), notes)
            timings["layerwise_semantics"] = time.perf_counter() - t
            t = time.perf_counter()
            clouds = _attempt(metrics, "token_clouds", lambda: token_clouds(occ, ctx), notes)
            timings["token_clouds"] = time.perf_counter() - t
            if _ok(clouds):
                metrics["token_clouds"] = _clouds_metrics(clouds)
        model = None                                                    # release the replayed model

        failed: List[Dict[str, str]] = []                               # failures outside the metrics tree
        resplits = {"bank": bank.diagnostics["forward_resplit"]}
        if _ok(lens):
            resplits["logit_lens"] = lens["forward_resplit"]
        if occ is not None and "forward_resplit" in occ:
            resplits["occurrences"] = occ["forward_resplit"]
        n_split = sum(r["resplit_batches"] for r in resplits.values())
        n_skip = sum(r["skipped_windows"] for r in resplits.values())
        n_oom = sum(r.get("oom_resplits", 0) for r in resplits.values())
        metrics["forward_resplits"] = {"resplit_batches": n_split, "oom_resplits": n_oom, "skipped_windows": n_skip,
                                       "by_step": resplits}
        if n_split or n_skip:
            detail = ", ".join(f"{step} {r['resplit_batches']}/{r['skipped_windows']}" for step, r in resplits.items())
            notes.append(f"{n_split} batches were re-run in smaller sub-batches because the model's exact-symmetry "
                         f"check failed under CPU BLAS summation order; {n_skip} windows were skipped "
                         f"(re-split batches/skipped windows: {detail})")
        if n_oom:
            detail = ", ".join(f"{step} {r.get('oom_resplits', 0)}" for step, r in resplits.items())
            notes.append(f"{n_oom} batches ran out of {ctx.device} memory and were re-run in halves ({detail}); "
                         "the per-window results do not depend on the batch composition")
        for step, r in resplits.items():
            if r["skipped_windows"]:
                failed.append({"step": f"{step} forward ({r['skipped_windows']} single windows skipped)",
                               "error": f"ValueError: projected covariance {_SYMMETRY_FAILURE}"})

        t = time.perf_counter()
        figure_jobs: List[Tuple[str, Any, Callable[[], List[Path]]]] = [
            ("logit_lens", lens, lambda: _plot_logit_lens(ctx, lens)),
            ("geometry", metrics["geometry"], lambda: _plot_geometry(ctx, metrics["geometry"])),
            ("cka", metrics["cka"], lambda: _plot_cka(ctx, metrics["cka"])),
            ("contextuality", metrics["contextuality"], lambda: _plot_contextuality(ctx, metrics["contextuality"])),
            ("layerwise_semantics", sem, lambda: _plot_layerwise_semantics(ctx, sem)),
            ("token_clouds", clouds, lambda: _plot_token_clouds(ctx, clouds)),
            ("sigma", metrics["sigma"], lambda: _plot_sigma(ctx, metrics["sigma"])),
        ]
        for name, data, plot in figure_jobs:
            if not _ok(data):
                continue
            if name == "token_clouds" and not data["words"]:
                notes.append("no ambiguous word has enough occurrences for a token-cloud figure")
                continue
            try:
                res.figures += plot()
            except Exception as exc:  # recorded, not swallowed
                notes.append(f"figure contextual/{name} failed: {type(exc).__name__}: {exc}")
                metrics.setdefault("figure_errors", {})[name] = traceback.format_exc()[-3000:]
                failed.append({"step": f"figure {name}", "error": f"{type(exc).__name__}: {exc}"})
        if metrics["sigma"].get("status") == "skipped":
            notes.append("no contextual/sigma figure: Σ does not move in context")
        for name, data, render in (("logit_lens", lens, lambda: _lens_table(ctx, lens)),
                                   ("layerwise_semantics", sem, lambda: _semantics_table(ctx, sem)),
                                   ("token_clouds", clouds, lambda: _clouds_table(clouds))):
            if _ok(data) and (name != "token_clouds" or data["words"]):
                try:
                    text = render()
                    write_table(ctx, _SECTION, name, text)
                except Exception as exc:  # recorded, not swallowed
                    notes.append(f"table contextual/{name} failed: {type(exc).__name__}: {exc}")
                    failed.append({"step": f"table {name}", "error": f"{type(exc).__name__}: {exc}"})
                else:
                    res.tables[name] = text
        timings["figures_tables"] = time.perf_counter() - t
        tree = _failed_steps(metrics)
        for rec in tree:                                                # top-level steps noted when caught
            if "." in rec["step"] or "[" in rec["step"]:
                notes.append(f"contextual {rec['step']} failed: {rec['error']}")
        metrics["failed_steps"] = tree + failed
        _headline(res, lens, metrics["cka"], metrics["contextuality"], sem, bank)
    finally:
        torch.set_num_threads(previous_threads)
        model = None                                                    # release the replayed model on any exit
        if torch.device(ctx.device).type == "cuda":
            torch.cuda.empty_cache()
    res.metrics = metrics
    res.seconds = time.perf_counter() - t_start
    return res
