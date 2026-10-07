r"""Realized-frame routing covariances and the hidden temperature of the stored variances (v2:872).

Routing covariance gap
----------------------
On the projected routes the router reads every belief in the reference coordinates of its realized
frame :math:`U_i = \exp(X_{x_i})\exp(Y_i)` (the ``relational_frame`` of ``VFEModel.forward_beliefs``,
RoPE folded when present). With the stack-entry covariance :math:`\Sigma_i` (the stored prior
:math:`\Sigma_{p,x_i}` for per-token encoding, the materialized one for projected canonical content),

.. math::
    C_i = U_i^{-1}\Sigma_i U_i^{-\top}, \qquad G_i^{-1} = (U_i^\top U_i)^{-1} = U_i^{-1}U_i^{-\top},

so :math:`C_i = c\,G_i^{-1}` exactly when :math:`\Sigma_i = c I`. The common scale is fitted by
relative least squares over the sampled pairs,

.. math::
    c = \arg\min_c \sum_i \frac{\|C_i - c G_i^{-1}\|_F^2}{\|C_i\|_F^2}
      = \frac{\sum_i \langle C_i, G_i^{-1}\rangle_F / \|C_i\|_F^2}{\sum_i \|G_i^{-1}\|_F^2 / \|C_i\|_F^2},

and each pair has the relative Frobenius gap :math:`g_i = \|C_i - c G_i^{-1}\|_F / \|C_i\|_F` and the
log-determinant gap :math:`\ell_i = |\log\det C_i - \log\det(c G_i^{-1})| / d`. Because
:math:`\det C_i = \det\Sigma_i / \det(U_i)^2`, the frames cancel in :math:`\ell_i`, which equals
:math:`|d^{-1}\log\det\Sigma_i - \log c|`: the log-det measure sees only the stored per-token scale,
the Frobenius measure also sees how the frame shapes :math:`C_i`. With block-diagonal frames and
diagonal :math:`\Sigma_i` both are also reported per frame block (:math:`d = d_b`, same :math:`c`).

Pairs (token type, position) are sampled two ways. ``occurrence``: every input position of evenly
spaced evaluation windows of the split (types weighted by their frequency in the stream, positions
uniform), the population the router actually sees; this is the headline. ``uniform_types``: synthetic
windows whose tokens are drawn uniformly from the distinct types of the split stream (seeded), so
each type counts equally. The frame and covariance at a position depend only on (type, position)
when both are held fixed through the stack (covariance passthrough); the result records whether the
stack kept the covariance fixed on a probe window, and otherwise :math:`C_i` is the layer-0 routing
covariance only.

Hidden temperature
------------------
A common rescaling :math:`\Sigma \to c\Sigma` of every covariance rescales the Mahalanobis term of the
Gaussian KL routing cost by :math:`1/c`. The stored log-variances start at
:math:`\log\sigma^2_{\mathrm{init}}` (``cfg.sigma_init``) and are moved by training (weight decay
pulls them toward unit variance), so the factor by which that term was sharpened is

.. math::
    \kappa = \frac{\sigma^2_{\mathrm{init}}}{\overline{\sigma^2}}, \qquad
    \overline{\sigma^2} = \frac{1}{VK}\sum_{v,k}\sigma^2_{v,k},

the ratio of the mean stored variance at initialization to the final one ("sharpened that term about
14-fold" in v2:872). Its per-coordinate distribution :math:`\sigma^2_{\mathrm{init}}/\sigma^2_{v,k}`,
the inverse-variance mean :math:`\sigma^2_{\mathrm{init}}\,\overline{1/\sigma^2}` and the
occurrence-weighted mean over the split stream are reported beside it, with the variance summary
quoted in v2:872 (mean, standard deviation, fraction within 10% of unit variance).
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

OCCURRENCE_WINDOWS: int = 64            # evaluation windows of the occurrence-weighted sample
UNIFORM_WINDOWS: int = 64               # synthetic windows of the uniform-over-types sample
SAMPLE_SEED: int = 0                    # generator seed of the uniform-over-types sample
FRAME_BATCH_WINDOWS: int = 8            # windows per frame construction
PAIR_CHUNK_SCALARS: int = 1 << 24       # float64 scalars per dense-covariance work chunk (128 MB)
PASSTHROUGH_TOL: float = 1e-6           # max relative covariance change through the stack for "fixed"
GAP_QUANTILES: Tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)
FACTOR_QUANTILES: Tuple[float, ...] = (0.01, 0.1, 0.5, 0.9, 0.99)
HEADLINE_SAMPLE: str = "occurrence"


# ----------------------------------------------------------------------------------------------
# Applicability and frames
# ----------------------------------------------------------------------------------------------

def gap_applicability(model: torch.nn.Module) -> Optional[str]:
    r"""``None`` when the realized-frame routing covariance is defined for ``model``, else the reason."""
    if not hasattr(model.prior_bank, "sigma_log_embed"):
        return "the prior bank has no stored log-variance table"
    if any(getattr(op, "mixed_heads", False) for op in getattr(model, "relational_operators_by_layer", ()) or ()):
        return "heads of different sizes (a mixed so_n irrep tower) are not supported by this analysis yet"
    return None


@torch.no_grad()
def routing_frame_and_covariance(
    model:     torch.nn.Module,
    token_ids: torch.Tensor,                      # (B, N) integer token ids on the model's device
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    r"""``(U^{-1} blocks (B, N, Hb, d, d), Sigma (B, N, K) or (B, N, K, K), block_diagonal)``.

    The construction is ``forward_beliefs``' own for the projected routes: encode, positional right
    factor, the shared flat transport, then the relational frame
    ``_canonical_frame_context(transport, fold_rope=True, compact=True)``
    (``VFEModel._relational_frames``). A dense
    frame is split into its head blocks, as ``relational_score`` does (which rejects a frame with
    off-block entries). Float32 frames outside autocast, as the model computes them.
    """
    device = token_ids.device
    with torch.autocast(device_type=device.type, enabled=False):
        beliefs = model.prior_bank.encode(token_ids)
        beliefs = beliefs._replace(phi=model._apply_pos_phi(beliefs.phi),
                                   right_phi=model._pos_phi_right(beliefs.phi))
        rope = model._rope_rotation(int(token_ids.shape[1]), device)
        _, frame, _ = model._relational_frames(beliefs, rope)
    inverse = frame.inverse
    block_diagonal = bool(frame.block_diagonal)
    if not block_diagonal:                                                    # (B, N, K, K) dense layout
        dims = list(model.relational_operators.irrep_dims)
        H, d = len(dims), dims[0]
        mask = torch.block_diag(*(torch.ones(d, d, dtype=torch.bool, device=inverse.device) for _ in range(H)))
        if H > 1 and bool((inverse[..., ~mask] != 0).any()):
            inverse = inverse.unsqueeze(-3)                                   # one dense block (B, N, 1, K, K)
        else:
            inverse = torch.stack([inverse[..., h*d:(h+1)*d, h*d:(h+1)*d] for h in range(H)], dim=-3)
            block_diagonal = True
    return inverse, beliefs.sigma, block_diagonal


# ----------------------------------------------------------------------------------------------
# Per-pair statistics
# ----------------------------------------------------------------------------------------------

def _pair_statistics_chunk(
    inverse: torch.Tensor,                        # (P, Hb, d, d) float64 U^{-1} blocks
    sigma:   torch.Tensor,                        # (P, K) or (P, K, K) float64
) -> Dict[str, torch.Tensor]:
    r"""Per pair and block: <C, G^{-1}>, ||G^{-1}||^2, ||C||^2, log det C, log det G^{-1}; whole-matrix sums."""
    P, Hb, d, _ = inverse.shape
    G = inverse @ inverse.transpose(-1, -2)                                   # (P, Hb, d, d) (U^T U)^{-1}
    if sigma.dim() == 2:                                                      # diagonal Sigma
        s = sigma.reshape(P, Hb, d)
        C = (inverse * s[:, :, None, :]) @ inverse.transpose(-1, -2)          # (P, Hb, d, d)
        cc_whole = None
    else:                                                                     # full Sigma: off-block C too
        blocks = sigma.reshape(P, Hb, d, Hb, d).permute(0, 1, 3, 2, 4)        # (P, a, b, d, d) Sigma_ab
        C_full = inverse[:, :, None] @ blocks @ inverse[:, None].transpose(-1, -2)   # (P, a, b, d, d)
        C = torch.diagonal(C_full, dim1=1, dim2=2).permute(0, 3, 1, 2)        # (P, Hb, d, d) diagonal blocks
        cc_whole = C_full.square().sum(dim=(1, 2, 3, 4))                      # (P,)
        dense = C_full.permute(0, 1, 3, 2, 4).reshape(P, Hb * d, Hb * d)      # (P, K, K)
    C = 0.5 * (C + C.transpose(-1, -2))
    out = {
        "ip_block": (C * G).sum(dim=(-1, -2)),                                # (P, Hb)
        "gg_block": G.square().sum(dim=(-1, -2)),
        "cc_block": C.square().sum(dim=(-1, -2)),
        "ldC_block": torch.linalg.slogdet(C)[1],
        "ldG_block": torch.linalg.slogdet(G)[1],
    }
    out["ip"] = out["ip_block"].sum(-1)                                       # G^{-1} is block-diagonal
    out["gg"] = out["gg_block"].sum(-1)
    out["cc"] = out["cc_block"].sum(-1) if cc_whole is None else cc_whole
    out["ldG"] = out["ldG_block"].sum(-1)
    out["ldC"] = (out["ldC_block"].sum(-1) if sigma.dim() == 2
                  else torch.linalg.slogdet(0.5 * (dense + dense.transpose(-1, -2)))[1])
    return out


def pair_statistics(
    inverse: torch.Tensor,                        # (B, N, Hb, d, d)
    sigma:   torch.Tensor,                        # (B, N, K) or (B, N, K, K)
    keep:    torch.Tensor,                        # (B, N) bool, the positions to keep
) -> Dict[str, np.ndarray]:
    r"""Float64 per-pair statistics of the kept positions, in chunks bounded by :data:`PAIR_CHUNK_SCALARS`."""
    Hb, d = int(inverse.shape[-3]), int(inverse.shape[-1])
    K = Hb * d
    inv = inverse[keep].double()                                              # (P, Hb, d, d)
    sig = sigma[keep].double()                                                # (P, K) or (P, K, K)
    per_pair = 4 * K * (K if sig.dim() == 3 else d)                          # live float64 scalars per pair
    step = max(1, PAIR_CHUNK_SCALARS // per_pair)
    parts: Dict[str, List[np.ndarray]] = {}
    for start in range(0, int(inv.shape[0]), step):
        chunk = _pair_statistics_chunk(inv[start:start + step], sig[start:start + step])
        for key, value in chunk.items():
            parts.setdefault(key, []).append(value.cpu().numpy())
    return {key: np.concatenate(value) for key, value in parts.items()}


def fit_common_scale(stats: Mapping[str, np.ndarray]) -> float:
    r"""The relative least-squares scale :math:`c` (module docstring)."""
    weight = 1.0 / stats["cc"]
    return float((stats["ip"] * weight).sum() / (stats["gg"] * weight).sum())


def gaps(stats: Mapping[str, np.ndarray], c: float, *, suffix: str = "") -> Tuple[np.ndarray, np.ndarray]:
    r"""``(relative Frobenius gap, log-det gap per dimension)`` of every pair (or block with ``suffix``)."""
    cc, ip, gg = stats["cc" + suffix], stats["ip" + suffix], stats["gg" + suffix]
    frob = np.sqrt(np.maximum(cc - 2.0 * c * ip + c * c * gg, 0.0) / cc)
    d = stats["_d_block"] if suffix else stats["_d"]
    logdet = np.abs(stats["ldC" + suffix] - (d * math.log(c) + stats["ldG" + suffix])) / d
    return frob, logdet


def _quantiles(values: np.ndarray, prefix: str, quantiles: Sequence[float] = GAP_QUANTILES) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return {f"{prefix}_n": 0}
    row: Dict[str, float] = {f"{prefix}_n": int(values.size), f"{prefix}_mean": float(values.mean()),
                             f"{prefix}_max": float(values.max())}
    for q, value in zip(quantiles, np.quantile(values, quantiles)):
        row[f"{prefix}_q{100 * q:g}".replace(".", "p")] = float(value)
    return row


# ----------------------------------------------------------------------------------------------
# Samples
# ----------------------------------------------------------------------------------------------

def _window_batches(dataset: Any, indices: np.ndarray, batch: int) -> "Any":
    from vfe3.posthoc.evaluate import iter_batches, window_start

    stream = getattr(dataset, "tokens", None)
    for windows, inputs, _ in iter_batches(dataset, indices, batch):
        N = inputs.shape[1]
        starts = np.array([window_start(dataset, int(w)) for w in windows], dtype=np.int64)
        keep = (torch.ones(inputs.shape, dtype=torch.bool) if stream is None else
                torch.as_tensor(starts[:, None] + np.arange(N)[None, :] < int(stream.numel())))   # (B, N)
        yield inputs, keep


def _uniform_batches(types: np.ndarray, n_windows: int, seq_len: int, batch: int, seed: int) -> "Any":
    rng = np.random.default_rng(seed)
    drawn = types[rng.integers(0, types.size, size=(n_windows, seq_len))]    # (W, N)
    for start in range(0, n_windows, batch):
        inputs = torch.as_tensor(drawn[start:start + batch], dtype=torch.long)
        yield inputs, torch.ones(inputs.shape, dtype=torch.bool)


def _sample_statistics(
    model:   torch.nn.Module,
    batches: Any,
    device:  torch.device,
) -> Tuple[Dict[str, np.ndarray], bool]:
    parts: Dict[str, List[np.ndarray]] = {}
    block_diagonal = True
    for inputs, keep in batches:
        inverse, sigma, block_diagonal = routing_frame_and_covariance(model, inputs.to(device))
        keep_device = keep.to(inverse.device)
        stats = pair_statistics(inverse, sigma, keep_device)
        stats["type"] = inputs[keep].numpy().astype(np.int64)
        stats["position"] = np.nonzero(keep.numpy())[1].astype(np.int64)
        for key, value in stats.items():
            parts.setdefault(key, []).append(value)
        d_block = int(inverse.shape[-1])
        n_blocks = int(inverse.shape[-3])
    out = {key: np.concatenate(value) for key, value in parts.items()}
    out["_d_block"] = float(d_block)
    out["_d"] = float(d_block * n_blocks)
    return out, block_diagonal


def summarize_sample(stats: Mapping[str, np.ndarray], c: float, *, block_diagonal: bool) -> Dict[str, Any]:
    r"""Gap quantiles (whole matrix and per block) of one sample at the common scale ``c``."""
    frob, logdet = gaps(stats, c)
    summary: Dict[str, Any] = {
        "n_pairs": int(frob.size), "n_types": int(np.unique(stats["type"]).size),
        "n_positions": int(np.unique(stats["position"]).size),
        **_quantiles(frob, "frobenius_gap"), **_quantiles(logdet, "logdet_gap"),
        "c_logdet": float(math.exp(np.mean((stats["ldC"] - stats["ldG"]) / stats["_d"]))),
    }
    if block_diagonal and stats["ip_block"].ndim == 2 and stats["ip_block"].shape[1] > 1:
        rows = []
        for b in range(stats["ip_block"].shape[1]):
            block = {key.replace("_block", "") + "_b": value[:, b] for key, value in stats.items()
                     if key.endswith("_block") and not key.startswith("_")}
            block["_d_block"] = stats["_d_block"]
            block["_d"] = stats["_d"]
            f_b, l_b = gaps(block, c, suffix="_b")
            rows.append({"block": b, **_quantiles(f_b, "frobenius_gap"), **_quantiles(l_b, "logdet_gap")})
        summary["per_block"] = rows
    summary["arrays"] = {"frobenius_gap": frob, "logdet_gap": logdet,
                         "type": stats["type"], "position": stats["position"]}
    return summary


@torch.no_grad()
def _stack_covariance_change(model: torch.nn.Module, inputs: torch.Tensor) -> Optional[float]:
    r"""Max relative change of the belief covariance through the stack on ``inputs`` (None if uncaptured)."""
    capture: Dict[str, Any] = {"diagnostic": {}}
    model.forward_beliefs(inputs, capture=capture)
    diagnostic = capture["diagnostic"]
    initial = diagnostic.get("initial_belief")
    outputs = diagnostic.get("layer_outputs") or []
    if initial is None or not outputs:
        return None
    base = initial.sigma.double()
    scale = float(base.abs().max()) or 1.0
    return max(float((layer.sigma.double() - base).abs().max()) / scale for layer in outputs[:-1]) \
        if len(outputs) > 1 else 0.0


def routing_covariance_gap(
    model:   torch.nn.Module,
    dataset: Any,

    *,
    stream_tokens:     Optional[torch.Tensor],
    device:            torch.device,
    n_windows:         int = OCCURRENCE_WINDOWS,
    n_uniform_windows: int = UNIFORM_WINDOWS,
    seed:              int = SAMPLE_SEED,
    batch_windows:     int = FRAME_BATCH_WINDOWS,
) -> Dict[str, Any]:
    r"""The v2:872 gap for the ``occurrence`` and ``uniform_types`` samples at one fitted scale each."""
    from vfe3.posthoc.streams import evenly_spaced

    reason = gap_applicability(model)
    if reason is not None:
        return {"status": "not_applicable", "reason": reason}
    indices = evenly_spaced(len(dataset), n_windows)
    samples: Dict[str, Any] = {}
    occurrence, block_diagonal = _sample_statistics(model, _window_batches(dataset, indices, batch_windows), device)
    raw = {"occurrence": occurrence}
    seq_len = int(getattr(dataset, "seq_len", model.cfg.max_seq_len))
    if n_uniform_windows > 0:
        stream = stream_tokens if stream_tokens is not None else getattr(dataset, "tokens", None)
        types = (np.unique(stream.reshape(-1).cpu().numpy()) if stream is not None
                 else np.arange(int(model.cfg.vocab_size)))
        raw["uniform_types"], _ = _sample_statistics(
            model, _uniform_batches(types, n_uniform_windows, seq_len, batch_windows, seed), device)
    for name, stats in raw.items():
        c = fit_common_scale(stats)
        samples[name] = {"c": c, **summarize_sample(stats, c, block_diagonal=block_diagonal)}
    probe = next(_window_batches(dataset, indices[:1], 1))[0].to(device)
    change = _stack_covariance_change(model, probe)
    fixed = change is not None and change <= PASSTHROUGH_TOL
    head = samples[HEADLINE_SAMPLE]
    return {
        "status": "ok", "headline_sample": HEADLINE_SAMPLE, "c": head["c"],
        "median_frobenius_gap": head.get("frobenius_gap_q50"), "median_logdet_gap": head.get("logdet_gap_q50"),
        "samples": samples, "block_diagonal": block_diagonal,
        "n_blocks": int(occurrence["ip_block"].shape[1]), "block_dim": int(occurrence["_d_block"]),
        "stack_covariance_max_rel_change": change, "covariance_fixed_through_stack": fixed,
        "scope": ("C_i depends only on (type, position): covariance and frames are fixed through the stack"
                  if fixed else "the stack changes the covariance: C_i is the layer-0 routing covariance only"),
        "sampling": {"occurrence": f"every input position of {indices.size} evenly spaced evaluation windows "
                                   "(types by stream frequency, positions uniform)",
                     "uniform_types": f"{n_uniform_windows} synthetic windows, types uniform over the distinct "
                                      f"types of the split stream (seed {seed})"},
        "definitions": {
            "c": "argmin_c sum_i ||C_i - c (U_i^T U_i)^{-1}||_F^2 / ||C_i||_F^2",
            "frobenius_gap": "||C_i - c (U_i^T U_i)^{-1}||_F / ||C_i||_F",
            "logdet_gap": "|log det C_i - log det(c (U_i^T U_i)^{-1})| / d = |log det(Sigma_i)/d - log c| "
                          "(frames cancel)",
            "c_logdet": "exp(mean_i (log det C_i - log det (U_i^T U_i)^{-1}) / K), the log-det-optimal scale"},
    }


# ----------------------------------------------------------------------------------------------
# Hidden temperature
# ----------------------------------------------------------------------------------------------

@torch.no_grad()
def hidden_temperature(
    model:       torch.nn.Module,

    *,
    type_counts: Optional[np.ndarray] = None,     # (V,) occurrences of each type in a stream
) -> Dict[str, Any]:
    r"""The v2:872 factor :math:`\kappa = \sigma^2_{\rm init}/\overline{\sigma^2}` and its distribution."""
    from vfe3.numerics import bounded_variance_from_log

    if getattr(model.cfg, "predictive_readout", "kernel") == "minimal_agent":
        # The factor is the kernel route's 1/c Mahalanobis scaling; the minimal agent's learned
        # channel noise n_h breaks that scaling (audit 2026-10-07 F85).
        return {"status": "not_applicable",
                "reason": "predictive_readout='minimal_agent': the learned channel noise n_h breaks the kernel "
                          "route's 1/c scaling that defines the hidden temperature"}
    pb = model.prior_bank
    if not hasattr(pb, "sigma_log_embed"):
        return {"status": "not_applicable", "reason": "the prior bank has no stored log-variance table"}
    sigma_init = getattr(model.cfg, "sigma_init", None)
    if sigma_init is None or not float(sigma_init) > 0:
        return {"status": "not_applicable", "reason": f"cfg.sigma_init={sigma_init!r} is not a positive variance"}
    eps = float(getattr(pb, "eps", 1e-6))
    variance = bounded_variance_from_log(pb._prior_sigma_log_table().detach().double(), eps=eps)   # (V, K)
    initial = float(bounded_variance_from_log(torch.tensor(math.log(float(sigma_init)), dtype=torch.float64),
                                              eps=eps))
    var = variance.cpu().numpy()
    factor = initial / var                                                    # (V, K)
    out: Dict[str, Any] = {
        "status": "ok", "sigma_init": float(sigma_init), "initial_variance": initial,
        "factor": initial / float(var.mean()),
        "factor_inverse_variance_mean": initial * float((1.0 / var).mean()),
        "variance_mean": float(var.mean()), "variance_sd": float(var.std()),
        "variance_min": float(var.min()), "variance_max": float(var.max()),
        "fraction_within_10pct_of_unit": float((np.abs(var - 1.0) <= 0.1).mean()),
        "n_entries": int(var.size), "table_shape": list(var.shape),
        **_quantiles(factor.reshape(-1), "per_coordinate_factor", FACTOR_QUANTILES),
        "definition": ("kappa = sigma_init^2 / mean_{v,k} sigma^2_{v,k}: a common rescaling Sigma -> c Sigma "
                       "rescales the Mahalanobis term of the routing KL by 1/c, so moving the stored variances "
                       "from their initial value to their final mean sharpened that term by kappa "
                       "(v2:872 'sharpened that term about 14-fold'); per coordinate sigma_init^2 / sigma^2_{v,k}"),
        "arrays": {"per_coordinate_factor": factor.reshape(-1)},
    }
    if type_counts is not None and np.asarray(type_counts).sum() > 0:
        counts = np.asarray(type_counts, dtype=np.float64)[: var.shape[0]]
        weighted = float((counts[:, None] * var[: counts.size]).sum() / (counts.sum() * var.shape[1]))
        out["variance_mean_occurrence_weighted"] = weighted
        out["factor_occurrence_weighted"] = initial / weighted
    return out


# ----------------------------------------------------------------------------------------------
# Driver and outputs
# ----------------------------------------------------------------------------------------------

def covariance_analysis(
    model:   torch.nn.Module,
    dataset: Optional[Any],

    *,
    stream_tokens:     Optional[torch.Tensor],
    device:            torch.device,
    n_windows:         int = OCCURRENCE_WINDOWS,
    n_uniform_windows: int = UNIFORM_WINDOWS,
) -> Dict[str, Any]:
    r"""Both v2:872 analyses; each part is isolated (a failure is recorded, the other still runs)."""
    result: Dict[str, Any] = {}
    counts = None
    if stream_tokens is not None:
        counts = np.bincount(stream_tokens.reshape(-1).cpu().numpy().astype(np.int64),
                             minlength=int(model.cfg.vocab_size))
    try:
        result["hidden_temperature"] = hidden_temperature(model, type_counts=counts)
    except Exception as exc:
        result["hidden_temperature"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    if dataset is None:
        result["routing_covariance"] = {"status": "skipped", "reason": "no evaluation split for the sample"}
    else:
        try:
            result["routing_covariance"] = routing_covariance_gap(
                model, dataset, stream_tokens=stream_tokens, device=device, n_windows=n_windows,
                n_uniform_windows=n_uniform_windows)
        except Exception as exc:
            result["routing_covariance"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    statuses = {part.get("status") for part in result.values()}
    result["status"] = "ok" if "ok" in statuses else sorted(statuses)[0]
    return result


def without_arrays(result: Any) -> Any:
    r"""A copy of ``result`` without the per-pair ``arrays`` entries (for ``posthoc.json``)."""
    if isinstance(result, Mapping):
        return {k: without_arrays(v) for k, v in result.items() if k != "arrays"}
    if isinstance(result, list):
        return [without_arrays(v) for v in result]
    return result


def covariance_table(result: Mapping[str, Any]) -> str:
    r"""Markdown of the v2:872 numbers at one set of weights."""
    def num(value: Any, spec: str = ".4g") -> str:
        return "–" if value is None or not np.isfinite(float(value)) else f"{float(value):{spec}}"

    lines: List[str] = []
    rc = result.get("routing_covariance") or {}
    if rc.get("status") == "ok":
        lines += ["| Sample | Pairs | Types | c | Frobenius gap median [q10, q90] | log-det gap median [q10, q90] | "
                  "c (log-det optimum) |", "|---|---|---|---|---|---|---|"]
        for name, s in rc["samples"].items():
            label = name + (" (headline)" if name == rc.get("headline_sample") else "")
            lines.append(f"| {label} | {s['n_pairs']:,} | {s['n_types']:,} | {num(s['c'])} | "
                         f"{num(s.get('frobenius_gap_q50'))} [{num(s.get('frobenius_gap_q10'))}, "
                         f"{num(s.get('frobenius_gap_q90'))}] | {num(s.get('logdet_gap_q50'))} "
                         f"[{num(s.get('logdet_gap_q10'))}, {num(s.get('logdet_gap_q90'))}] | "
                         f"{num(s.get('c_logdet'))} |")
        head = rc["samples"].get(rc.get("headline_sample"), {})
        if head.get("per_block"):
            lines += ["", "| Block | Frobenius gap median | log-det gap median |", "|---|---|---|"]
            lines += [f"| {r['block']} | {num(r.get('frobenius_gap_q50'))} | {num(r.get('logdet_gap_q50'))} |"
                      for r in head["per_block"]]
        defs = rc.get("definitions", {})
        lines += ["", f"Frobenius gap = {defs.get('frobenius_gap')}; log-det gap = {defs.get('logdet_gap')}; "
                      f"c = {defs.get('c')}. {rc.get('scope')}. Sampling: "
                      + "; ".join(f"{k}: {v}" for k, v in rc.get("sampling", {}).items()) + "."]
    else:
        lines.append(f"Routing covariance gap {rc.get('status')}: {rc.get('reason') or rc.get('error')}.")
    ht = result.get("hidden_temperature") or {}
    lines.append("")
    if ht.get("status") == "ok":
        lines += ["| Hidden temperature | Value |", "|---|---|",
                  f"| sigma_init^2 / mean sigma^2 (headline) | {num(ht['factor'])} |",
                  f"| sigma_init^2 x mean(1/sigma^2) | {num(ht['factor_inverse_variance_mean'])} |",
                  f"| occurrence-weighted | {num(ht.get('factor_occurrence_weighted'))} |",
                  f"| per coordinate median [q01, q99] | {num(ht.get('per_coordinate_factor_q50'))} "
                  f"[{num(ht.get('per_coordinate_factor_q1'))}, {num(ht.get('per_coordinate_factor_q99'))}] |",
                  f"| stored variance mean +- sd | {num(ht['variance_mean'])} +- {num(ht['variance_sd'])} |",
                  f"| entries within 10% of unit variance | {100 * ht['fraction_within_10pct_of_unit']:.1f}% |",
                  "", f"Definition: {ht['definition']}."]
    else:
        lines.append(f"Hidden temperature {ht.get('status')}: {ht.get('reason') or ht.get('error')}.")
    lines.append(f"\nWeights: {result.get('checkpoint')}.")
    return "\n".join(lines) + "\n"


def write_covariance_outputs(result: Mapping[str, Any], out_dir: Path, *, figures: bool = True) -> List[Path]:
    r"""``tables/covariance_<stem>.md``, ``csv/covariance_blocks_<stem>.csv``, per-pair ``.npz`` and figure."""
    from vfe3.posthoc.figures import plot_covariance_gap

    out_dir = Path(out_dir)
    stem = str(result.get("stem") or "final")
    written: List[Path] = []
    table = out_dir / "tables" / f"covariance_{stem}.md"
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text(covariance_table(result), encoding="utf-8")
    written.append(table)
    rc = result.get("routing_covariance") or {}
    arrays: Dict[str, np.ndarray] = {}
    if rc.get("status") == "ok":
        rows = [dict(r, sample=name) for name, s in rc["samples"].items() for r in s.get("per_block", [])]
        if rows:
            path = out_dir / "csv" / f"covariance_blocks_{stem}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            columns = ["sample"] + [k for k in rows[0] if k != "sample"]
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns, restval="")
                writer.writeheader()
                writer.writerows(rows)
            written.append(path)
        for name, s in rc["samples"].items():
            for key, value in (s.get("arrays") or {}).items():
                arrays[f"{name}_{key}"] = np.asarray(value)
    ht = result.get("hidden_temperature") or {}
    if ht.get("status") == "ok" and ht.get("arrays"):
        arrays["per_coordinate_factor"] = np.asarray(ht["arrays"]["per_coordinate_factor"], dtype=np.float32)
    if arrays:
        path = out_dir / "csv" / f"covariance_pairs_{stem}.npz"
        np.savez_compressed(path, **arrays)
        written.append(path)
    if figures and (rc.get("status") == "ok" or ht.get("status") == "ok"):
        written += plot_covariance_gap(result, out_dir / "figures", f"v2_872_covariance_{stem}")
    return written
