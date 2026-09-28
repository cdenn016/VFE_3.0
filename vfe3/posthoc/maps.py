r"""Invariants of the trained relational interface maps (S1 panel B, S2 panel A of GL(K)_attention_v2).

Each layer holds, per head :math:`h`, the comparison maps :math:`S_{Q,h}, S_{K,h}` (``raw_score_q``,
``raw_score_k``; :math:`r_h \times w`), the content maps :math:`T_h` (``T``; :math:`s_h \times w`) and
:math:`O_h` (``O``; :math:`K_{out} \times s_h`), with :math:`w = K` under full input scope and
:math:`w = d` under head scope. The frame blocks are the :math:`B = K/d` coordinate ranges
:math:`[bd, (b+1)d)`; on the evaluated route :math:`H = B` and :math:`r_h = s_h = d`.

Theorem ``thm:reference_moments`` (3) leaves every cost unchanged under
:math:`(S_{Q,h}, S_{K,h}) \mapsto (A S_{Q,h}, A S_{K,h})`, :math:`A \in \mathrm{GL}(r_h)`, and every
message unchanged under :math:`(T_h, O_h) \mapsto (B T_h, O_h B^{-1})`. The invariants reported are

* the relative gains, :math:`g = \sqrt{\operatorname{eig}((S_K S_K^\top)^{-1} S_Q S_Q^\top)}`;
* the principal angles between the row spaces of :math:`S_{Q,h}` and :math:`S_{K,h}`;
* :math:`\sigma_{\max}(O_h T_h)` and :math:`\|O_h T_h\|_F` of the realized content map;

together with the motion along the symmetry directions, which gradient flow conserves and adaptive
optimizers without weight decay do not: the common scale
:math:`\sqrt{\operatorname{tr}(S_Q S_Q^\top + S_K S_K^\top)/(2r)}` and the drifts from initialization
:math:`\|S_Q S_Q^\top + S_K S_K^\top - 2I\|_F` (orthonormal-row initialization) and
:math:`\|T T^\top - O^\top O - (1 - s_0^2) I\|_F` (block-placed :math:`T = I`, :math:`O = s_0 I`).

:func:`head_block_norms` gives the S2 panel A matrices: the Frobenius norm of each head's map restricted
to each frame block's coordinates (input columns for :math:`S_Q, S_K, T`; output rows for :math:`O`).

Everything is computed in float64 on the CPU from a state dict, a checkpoint bundle, or a map snapshot
``{"step", "state"}``; :func:`snapshot_curves` turns a directory of ``step_*.pt`` snapshots into
long-format rows over training.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch

MAP_NAMES: Tuple[str, ...] = ("S_Q", "S_K", "T", "O")
_LEAF_BY_MAP = {"S_Q": "raw_score_q", "S_K": "raw_score_k", "T": "T", "O": "O"}
_LAYER_KEY = re.compile(r"^relational_operators(?:_extra\.(\d+))?\.(raw_score_q|raw_score_k|T|O)$")
_SNAPSHOT_FILE = re.compile(r"^step_(\d+)\.pt$")


# ----------------------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------------------

def unwrap_state(source: Mapping[str, Any]) -> Tuple[Mapping[str, torch.Tensor], Optional[int]]:
    r"""Return ``(state, step)`` from a state dict, a bundle with ``model_state``, or a snapshot."""
    if "model_state" in source and isinstance(source["model_state"], Mapping):
        step = source.get("step")
        return source["model_state"], (int(step) if isinstance(step, int) else None)
    if "state" in source and isinstance(source["state"], Mapping):
        step = source.get("step")
        return source["state"], (int(step) if isinstance(step, int) else None)
    return source, None


def load_map_state(source: Union[str, Path, Mapping[str, Any]]) -> Tuple[Mapping[str, torch.Tensor], Optional[int]]:
    r"""Load a map-bearing state from a path (``mmap``; weights only) or unwrap a loaded mapping."""
    if isinstance(source, (str, Path)):
        loaded = torch.load(Path(source), map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(loaded, Mapping):
            raise ValueError(f"{source} does not hold a mapping")
        return unwrap_state(loaded)
    return unwrap_state(source)


def relational_maps(state: Mapping[str, torch.Tensor]) -> Dict[int, Dict[str, torch.Tensor]]:
    r"""``{layer: {"S_Q", "S_K", "T", "O"}}`` float64 CPU tensors from the relational state keys.

    Layer 0 is ``relational_operators.*`` and layer :math:`\ell \ge 1` is
    ``relational_operators_extra.{l-1}.*``. Layers without content maps (identity content) omit
    ``T`` and ``O``.
    """
    layers: Dict[int, Dict[str, torch.Tensor]] = {}
    inverse = {leaf: name for name, leaf in _LEAF_BY_MAP.items()}
    for key, value in state.items():
        match = _LAYER_KEY.match(key)
        if match is None:
            continue
        layer = 0 if match.group(1) is None else int(match.group(1)) + 1
        layers.setdefault(layer, {})[inverse[match.group(2)]] = value.detach().to("cpu", torch.float64)
    if not layers:
        raise ValueError("state holds no relational_operators maps")
    for layer, maps in layers.items():
        if "S_Q" not in maps or "S_K" not in maps:
            raise ValueError(f"layer {layer} lacks a comparison map")
    return dict(sorted(layers.items()))


def _frame_geometry(maps: Mapping[str, torch.Tensor], cfg: Any) -> Tuple[int, int, int]:
    r"""Return ``(H, d, B)``: heads, frame-block dimension, number of frame blocks."""
    H, r, _ = maps["S_Q"].shape
    K = int(getattr(cfg, "embed_dim", 0) or 0)
    if K <= 0:
        K = int(maps["O"].shape[1]) if "O" in maps else int(maps["S_Q"].shape[-1])
    d = int(r)                                        # r_h = d_b on the relational operators
    if K % d:
        raise ValueError(f"embed_dim {K} is not a multiple of the head rank {d}")
    return int(H), d, K // d


# ----------------------------------------------------------------------------------------------
# S2 panel A
# ----------------------------------------------------------------------------------------------

def _block_norms_input(M: torch.Tensor, d: int, B: int) -> np.ndarray:
    r"""(H, B) Frobenius norms of each head's (rows, w) map on the input columns of each block."""
    H, _, w = M.shape
    out = np.zeros((H, B))
    if w == d * B:
        blocks = M.reshape(H, M.shape[1], B, d)                      # (H, r, B, d)
        out[:] = blocks.square().sum(dim=(1, 3)).sqrt().numpy()
    elif w == d and H == B:                                          # head scope: head h reads block h
        out[np.arange(H), np.arange(H)] = M.square().sum(dim=(1, 2)).sqrt().numpy()
    else:
        raise ValueError(f"map input width {w} matches neither K={d * B} nor the block size {d}")
    return out


def _block_norms_output(O: torch.Tensor, d: int, B: int) -> np.ndarray:
    r"""(H, B) Frobenius norms of each head's (K_out, s) writer on the output rows of each block."""
    H, rows, _ = O.shape
    out = np.zeros((H, B))
    if rows == d * B:
        blocks = O.reshape(H, B, d, O.shape[-1])                     # (H, B, d, s)
        out[:] = blocks.square().sum(dim=(2, 3)).sqrt().numpy()
    elif rows == d and H == B:                                       # head scope: head h writes block h
        out[np.arange(H), np.arange(H)] = O.square().sum(dim=(1, 2)).sqrt().numpy()
    else:
        raise ValueError(f"output map height {rows} matches neither K={d * B} nor the block size {d}")
    return out


def head_block_norms(
    state: Mapping[str, torch.Tensor],
    cfg:   Any,
) -> Dict[int, Dict[str, np.ndarray]]:
    r"""``{layer: {"S_Q", "S_K", "T", "O"} -> (H, B)}`` block-restricted Frobenius norms.

    Entry :math:`(h, b)` is :math:`\|M_h|_b\|_F`, the norm of head :math:`h`'s map restricted to frame
    block :math:`b`'s coordinates: the input columns :math:`[bd, (b+1)d)` of :math:`S_{Q,h}`,
    :math:`S_{K,h}`, :math:`T_h`, and the output rows of :math:`O_h`.
    """
    state, _ = unwrap_state(state)
    out: Dict[int, Dict[str, np.ndarray]] = {}
    for layer, maps in relational_maps(state).items():
        H, d, B = _frame_geometry(maps, cfg)
        norms = {name: _block_norms_input(maps[name], d, B) for name in ("S_Q", "S_K", "T") if name in maps}
        if "O" in maps:
            norms["O"] = _block_norms_output(maps["O"], d, B)
        out[layer] = norms
    return out


def block_energy_shares(norms: np.ndarray) -> np.ndarray:
    r"""(H, B) energy shares :math:`\|M_h|_b\|_F^2 / \sum_{b'} \|M_h|_{b'}\|_F^2` (rows sum to one)."""
    energy = np.square(np.asarray(norms, dtype=np.float64))
    total = energy.sum(axis=1, keepdims=True)
    return np.divide(energy, total, out=np.full_like(energy, np.nan), where=total > 0)


# ----------------------------------------------------------------------------------------------
# S1 panel B
# ----------------------------------------------------------------------------------------------

def _output_init_scale(cfg: Any) -> float:
    r"""Initial content-writer scale :math:`s_0` (``relational_output_init``)."""
    if getattr(cfg, "relational_output_init", None) == "depth_scaled":
        return 1.0 / math.sqrt(2.0 * float(getattr(cfg, "n_layers", 1)))
    return 1.0


def relative_gains(S_Q: torch.Tensor, S_K: torch.Tensor) -> np.ndarray:
    r"""(r,) ascending :math:`\sqrt{\operatorname{eig}((S_K S_K^\top)^{-1} S_Q S_Q^\top)}`.

    Computed as the symmetric generalized problem :math:`L^{-1} A L^{-\top}` with
    :math:`S_K S_K^\top = L L^\top` (float64). Invariant under :math:`(S_Q, S_K) \mapsto (AS_Q, AS_K)`.
    """
    A = S_Q @ S_Q.T                                                   # (r, r)
    Bm = S_K @ S_K.T
    L = torch.linalg.cholesky(Bm)
    X = torch.linalg.solve_triangular(L, A, upper=False)
    C = torch.linalg.solve_triangular(L, X.T, upper=False)            # L^-1 A L^-T
    C = 0.5 * (C + C.T)
    return torch.linalg.eigvalsh(C).clamp_min(0.0).sqrt().numpy()


def principal_angles(S_Q: torch.Tensor, S_K: torch.Tensor) -> np.ndarray:
    r"""(r,) ascending principal angles (radians) between the row spaces of ``S_Q`` and ``S_K``."""
    Q_q, _ = torch.linalg.qr(S_Q.T, mode="reduced")                  # (w, r) orthonormal basis
    Q_k, _ = torch.linalg.qr(S_K.T, mode="reduced")
    cosines = torch.linalg.svdvals(Q_q.T @ Q_k).clamp(0.0, 1.0)       # descending cosines
    return torch.arccos(cosines).numpy()


def map_invariants(
    state: Mapping[str, torch.Tensor],
    cfg:   Any,
) -> List[Dict[str, float]]:
    r"""One row per (layer, head) with the S1 panel B quantities and the S2 panel A block shares.

    Columns: ``layer``, ``head``; Frobenius norms ``S_Q_fro``, ``S_K_fro``, ``T_fro``, ``O_fro``; the raw
    (basis-dependent) smallest singular values ``S_Q_sv_min``, ``S_K_sv_min``; the common scale
    ``scale_rms``; the relative gains ``gain_min``, ``gain_median``, ``gain_max``; the principal angles
    ``angle_min_deg``, ``angle_median_deg``, ``angle_max_deg``; ``M_sigma_max`` and ``M_fro`` of
    :math:`M_h = O_h T_h`; the conserved-quantity drifts ``conserved_S_drift``, ``conserved_TO_drift``;
    and per map the energy share on the head's own block and the largest block share
    (``{map}_own_block_share``, ``{map}_max_block_share``).
    """
    state, _ = unwrap_state(state)
    s0 = _output_init_scale(cfg)
    rows: List[Dict[str, float]] = []
    norms_by_layer = head_block_norms(state, cfg)
    for layer, maps in relational_maps(state).items():
        H, d, B = _frame_geometry(maps, cfg)
        shares = {name: block_energy_shares(norms) for name, norms in norms_by_layer[layer].items()}
        for h in range(H):
            S_Q, S_K = maps["S_Q"][h], maps["S_K"][h]                 # (r, w)
            r = S_Q.shape[0]
            eye_r = torch.eye(r, dtype=torch.float64)
            gains = relative_gains(S_Q, S_K)
            angles = np.degrees(principal_angles(S_Q, S_K))
            row: Dict[str, float] = {
                "layer": int(layer), "head": int(h),
                "S_Q_fro": float(S_Q.norm()), "S_K_fro": float(S_K.norm()),
                "S_Q_sv_min": float(torch.linalg.svdvals(S_Q).min()),
                "S_K_sv_min": float(torch.linalg.svdvals(S_K).min()),
                "scale_rms": float(math.sqrt(float((S_Q.square().sum() + S_K.square().sum()) / (2 * r)))),
                "gain_min": float(gains.min()), "gain_median": float(np.median(gains)),
                "gain_max": float(gains.max()),
                "angle_min_deg": float(angles.min()), "angle_median_deg": float(np.median(angles)),
                "angle_max_deg": float(angles.max()),
                "conserved_S_drift": float((S_Q @ S_Q.T + S_K @ S_K.T - 2.0 * eye_r).norm()),
            }
            if "T" in maps and "O" in maps:
                T, O = maps["T"][h], maps["O"][h]                     # (s, w), (K_out, s)
                s = T.shape[0]
                M = O @ T                                              # (K_out, w) realized content map
                row.update({
                    "T_fro": float(T.norm()), "O_fro": float(O.norm()),
                    "M_sigma_max": float(torch.linalg.svdvals(M).max()), "M_fro": float(M.norm()),
                    "conserved_TO_drift": float(
                        (T @ T.T - O.T @ O - (1.0 - s0 * s0) * torch.eye(s, dtype=torch.float64)).norm()),
                })
            else:
                row.update({"T_fro": float("nan"), "O_fro": float("nan"), "M_sigma_max": float("nan"),
                            "M_fro": float("nan"), "conserved_TO_drift": float("nan")})
            for name in MAP_NAMES:
                if name in shares:
                    own = shares[name][h, h] if h < B else float("nan")
                    row[f"{name}_own_block_share"] = float(own)
                    row[f"{name}_max_block_share"] = float(np.nanmax(shares[name][h]))
                else:
                    row[f"{name}_own_block_share"] = float("nan")
                    row[f"{name}_max_block_share"] = float("nan")
            rows.append(row)
    return rows


# ----------------------------------------------------------------------------------------------
# Over training
# ----------------------------------------------------------------------------------------------

def snapshot_files(snapshot_dir: Union[str, Path]) -> List[Tuple[int, Path]]:
    r"""``[(step, path)]`` of the ``step_<N>.pt`` files in ``snapshot_dir``, sorted by step."""
    directory = Path(snapshot_dir)
    if not directory.is_dir():
        return []
    found = []
    for path in directory.iterdir():
        match = _SNAPSHOT_FILE.match(path.name)
        if match is not None:
            found.append((int(match.group(1)), path))
    return sorted(found)


def snapshot_curves(
    snapshot_dir: Union[str, Path],
    cfg:          Any,
) -> List[Dict[str, float]]:
    r"""Long-format :func:`map_invariants` rows with a leading ``step`` for every snapshot.

    Each ``step_<N>.pt`` holds ``{"step": N, "state": {name: tensor}}``; the file name supplies the
    step when the payload does not.
    """
    rows: List[Dict[str, float]] = []
    for step, path in snapshot_files(snapshot_dir):
        state, payload_step = load_map_state(path)
        for row in map_invariants(state, cfg):
            rows.append({"step": int(payload_step if payload_step is not None else step), **row})
    return rows
