r"""Invariants of the trained relational interface maps (S1 panel B, S2 panel A of GL(K)_attention_v2).

Each layer holds, per head :math:`h`, the comparison maps :math:`S_{Q,h}, S_{K,h}` (``raw_score_q``,
``raw_score_k``; :math:`r_h \times w`), the content maps :math:`T_h` (``T``; :math:`s_h \times w`) and
:math:`O_h` (``O``; :math:`K_{out} \times s_h`), with :math:`w = K` under full input scope and
:math:`w = d` under head scope. The frame blocks are the :math:`B = K/d` coordinate ranges
:math:`[bd, (b+1)d)`; on the evaluated route :math:`H = B` and :math:`r_h = s_h = d`.

Theorem ``thm:reference_moments`` (3) leaves every cost unchanged under
:math:`(S_{Q,h}, S_{K,h}) \mapsto (A S_{Q,h}, A S_{K,h})`, :math:`A \in \mathrm{GL}(r_h)`, every
message unchanged under :math:`(T_h, O_h) \mapsto (B T_h, O_h B^{-1})`, and routing and means unchanged
under the common reference change :math:`(U_i, S, T_h, O_h) \mapsto (U_i a, S a, T_h a, a^{-1} O_h)`
(block-diagonal :math:`a` on the evaluated route; the moment-matched covariance floor is not invariant
under it). The quantities reported are

* the relative gains, :math:`g = \sqrt{\operatorname{eig}((S_K S_K^\top)^{-1} S_Q S_Q^\top)}`, and
  the principal angles between the row spaces of :math:`S_{Q,h}` and :math:`S_{K,h}`: invariant under
  :math:`A`, not under the reference change;
* :math:`\sigma_{\max}(O_h T_h)` and :math:`\|O_h T_h\|_F` of the realized content map: invariant under
  :math:`B`, not under the reference change (:math:`M_h \mapsto a^{-1} M_h a`);
* the ``link_*`` columns, similarity invariants of :math:`M_h = \hat O_h \hat T_h` read from the
  :math:`d \times d` matrix :math:`\hat T_h \hat O_h` (:func:`vfe3.metrics.content_link_head_invariants`):
  invariant under :math:`B` and the reference change, except the numerics certificate
  ``link_eigvec_condition``;

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

from vfe3.metrics import content_link_eigenvalues, content_link_head_invariants

MAP_NAMES: Tuple[str, ...] = ("S_Q", "S_K", "T", "O")
LINK_COLUMNS: Tuple[str, ...] = (
    "link_spectral_radius", "link_min_abs_eigenvalue", "link_mean_log_abs_eigenvalue", "link_unit_distance",
    "link_idempotency_defect", "link_trace", "link_trace_sq", "link_cross_trace", "link_sum_spectral_radius",
    "link_eigvec_condition")
_LEAF_BY_MAP = {"S_Q": "raw_score_q", "S_K": "raw_score_k", "T": "T", "O": "O"}
# content_link_edges factor maps (vfe3/model/edge_links.py), read under their own names.
EDGE_MAP_NAMES: Tuple[str, ...] = ("edge_receiver_map", "edge_site_basis", "edge_sender_map")
# predictive_readout='minimal_agent' (vfe3/model/minimal_agent.py): the channel log-noise, read under its own name.
MINIMAL_AGENT_NAMES: Tuple[str, ...] = ("channel_log_noise",)
_LAYER_KEY = re.compile(r"^relational_operators(?:_extra\.(\d+))?\."
                        r"(raw_score_q|raw_score_k|T|O|edge_receiver_map|edge_site_basis|edge_sender_map"
                        r"|channel_log_noise)$")
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


def relational_maps(
    state: Mapping[str, torch.Tensor],
    cfg:   Any = None,
) -> Dict[int, Dict[str, torch.Tensor]]:
    r"""``{layer: {"S_Q", "S_K", "T", "O"}}`` float64 CPU tensors from the relational state keys.

    Layer 0 is ``relational_operators.*`` and layer :math:`\ell \ge 1` is
    ``relational_operators_extra.{l-1}.*``. Layers without content maps (identity content) omit
    ``T`` and ``O``. With a ``cfg`` whose ``content_link_chart`` is not ``"affine"``, ``"O"`` is the
    effective receiving map :math:`O'` of that chart (:func:`vfe3.model.relational_operators.chart_receive_map`)
    and ``"O_stored"`` the stored factor. The ``content_link_edges`` factor maps, when stored, appear
    under their parameter names (:data:`EDGE_MAP_NAMES`). A minimal-agent layer (``channel_log_noise`` stored, no
    ``raw_score_k`` and no ``O``) ties the key to the content, so ``"S_K"`` is its ``T``; it has no ``"O"`` (the
    receiving map is derived from the receiver covariance) and carries ``"channel_log_noise"``.
    A kernel layer stored without ``raw_score_k`` (``tie_comparison_maps``) reports ``"S_K"`` as its ``"S_Q"``.
    """
    layers: Dict[int, Dict[str, torch.Tensor]] = {}
    inverse = {leaf: name for name, leaf in _LEAF_BY_MAP.items()}
    inverse.update({name: name for name in EDGE_MAP_NAMES + MINIMAL_AGENT_NAMES})
    for key, value in state.items():
        match = _LAYER_KEY.match(key)
        if match is None:
            continue
        layer = 0 if match.group(1) is None else int(match.group(1)) + 1
        layers.setdefault(layer, {})[inverse[match.group(2)]] = value.detach().to("cpu", torch.float64)
    if not layers:
        raise ValueError("state holds no relational_operators maps")
    for layer, maps in layers.items():
        if "channel_log_noise" in maps and "S_K" not in maps and "T" in maps:
            maps["S_K"] = maps["T"]                                     # minimal agent: tied key S_K = T
        elif "S_K" not in maps and "S_Q" in maps and "channel_log_noise" not in maps:
            maps["S_K"] = maps["S_Q"]                                   # tie_comparison_maps: S_K := S_Q
        if "S_Q" not in maps or "S_K" not in maps:
            raise ValueError(f"layer {layer} lacks a comparison map")
    chart = getattr(cfg, "content_link_chart", "affine")
    if chart != "affine":
        from vfe3.model.relational_operators import chart_receive_map

        for maps in layers.values():
            if "T" in maps and "O" in maps:
                maps["O_stored"] = maps["O"]
                maps["O"] = chart_receive_map(maps["T"], maps["O"], chart)
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
    for layer, maps in relational_maps(state, cfg).items():
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
    r"""Initial content-writer scale :math:`s_0` (1 since 2026-10-12; a historical record may carry the
    retired ``relational_output_init``)."""
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


def _ambient_content(T: torch.Tensor, O: torch.Tensor, d: int, B: int) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""(T_hat (H, d, K), O_hat (H, K, d)) from stored shapes; width K is full scope, width d head scope.

    Head scope places :math:`T_h` on the input columns and :math:`O_h` on the output rows of block
    :math:`h`, so :math:`M_h = \hat O_h \hat T_h` is the head's :math:`K \times K` content map in every scope.
    """
    H, K = T.shape[0], d * B
    if T.shape[-1] == K:
        T_hat = T
    elif T.shape[-1] == d and H == B:
        T_hat = T.new_zeros(H, d, K)
        for h in range(H):
            T_hat[h, :, h*d:(h+1)*d] = T[h]
    else:
        raise ValueError(f"content map width {T.shape[-1]} matches neither K={K} nor d={d}")
    if O.shape[1] == K:
        O_hat = O
    elif O.shape[1] == d and H == B:
        O_hat = O.new_zeros(H, K, d)
        for h in range(H):
            O_hat[h, h*d:(h+1)*d, :] = O[h]
    else:
        raise ValueError(f"output map height {O.shape[1]} matches neither K={K} nor d={d}")
    return T_hat, O_hat


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
    per map the energy share on the head's own block and the largest block share
    (``{map}_own_block_share``, ``{map}_max_block_share``); and the ``LINK_COLUMNS`` of
    :func:`vfe3.metrics.content_link_head_invariants`: the spectral radius, smallest modulus and mean
    log-modulus of the eigenvalues :math:`\lambda` of :math:`\hat T_h \hat O_h`, the unit distance
    :math:`\max|\lambda - 1|`, the idempotency defect :math:`\max|\lambda(\lambda - 1)|`, the traces
    :math:`\operatorname{Tr} M_h`, :math:`\operatorname{Tr} M_h^2`, :math:`\sum_{g \ne h} \operatorname{Tr} M_h M_g`,
    the per-layer :math:`\rho(\sum_g M_g)`, and the basis-dependent numerics certificate
    ``link_eigvec_condition`` :math:`\kappa_2(V_h)`. All ``link_*`` columns are NaN without content maps.

    A layer with ``content_link_edges`` (edge maps in the state) links edge :math:`j \to i` by
    :math:`G_{ij}\Pi_h = g_ie^{L}E_{ij}\Pi_h`, which depends on the context. Its content-map and ``link_*``
    columns are those of the shared generator :math:`e^{L}` (the :math:`B = 0` link), marked by the column
    ``link_shared_generator_only = 1.0``, and its rows add the
    Frobenius norms ``{edge map}_fro`` of the stored factor maps; the per-edge links themselves are
    data-dependent and are reported by the model diagnostics (``relational_edge_*``), not here.

    Under comparison-space RoPE (a comparison ``rope_mode``) the routing cost
    is :math:`\mathrm{KL}(R_i A a_i \Vert R_j A b_j)`, invariant only under the commutant of the
    rotations, so the comparison-pair charge :math:`S_Q S_Q^\top + S_K S_K^\top` is not conserved by
    gradient flow. ``conserved_S_drift`` is then NaN and the row carries ``comparison_rope = 1.0``;
    ``scale_rms`` (the trace charge, which survives) is kept, and the gains and angles are invariants
    of the larger GL(r) group, not of the model's symmetry (audit 2026-10-08, transformer-ml#5).
    """
    state, _ = unwrap_state(state)
    s0 = _output_init_scale(cfg)
    comparison_rope = getattr(cfg, "rope_mode", "none") in ("comparison", "comparison_invariant")
    rows: List[Dict[str, float]] = []
    norms_by_layer = head_block_norms(state, cfg)
    for layer, maps in relational_maps(state, cfg).items():
        H, d, B = _frame_geometry(maps, cfg)
        shares = {name: block_energy_shares(norms) for name, norms in norms_by_layer[layer].items()}
        link_rows = (content_link_head_invariants(*_ambient_content(maps["T"], maps["O"], d, B))
                     if "T" in maps and "O" in maps else None)
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
                    # The balancedness T T^T - O^T O of the affine factor flow; undefined in other charts.
                    "conserved_TO_drift": (float(
                        (T @ T.T - O.T @ O - (1.0 - s0 * s0) * torch.eye(s, dtype=torch.float64)).norm())
                        if "O_stored" not in maps else float("nan")),
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
            row.update(link_rows[h] if link_rows is not None else {c: float("nan") for c in LINK_COLUMNS})
            if "channel_log_noise" in maps:                           # minimal agent: S_K columns describe T
                # B = T C T^T + n_h I: (S_Q, T) -> (A S_Q, A T) is a symmetry only for orthogonal A, and a common
                # scale c only together with n_h -> c^2 n_h. The gains and angles (GL(r)-invariant, hence
                # O(r)-invariant) stay; the common scale is a signal-to-noise coordinate and S S^T + T T^T
                # has no conservation law, so those two columns are replaced by noise-unit singular values.
                n_h = float(maps["channel_log_noise"][h].exp())
                unit = math.sqrt(n_h)
                row["key_tied_to_content"] = 1.0
                row["channel_noise"] = n_h
                row["scale_rms"] = float("nan")
                row["conserved_S_drift"] = float("nan")
                row["scale_rms_noise_units"] = float(math.sqrt(float(
                    (S_Q.square().sum() + S_K.square().sum()) / (2 * r)))) / unit
                for name, mat in (("S_Q", S_Q), ("T", S_K)):
                    sv = torch.linalg.svdvals(mat) / unit
                    row[f"{name}_sv_noise_units_min"] = float(sv.min())
                    row[f"{name}_sv_noise_units_max"] = float(sv.max())
            if comparison_rope:                                       # RoPE breaks GL(r) to its commutant
                row["comparison_rope"] = 1.0
                row["conserved_S_drift"] = float("nan")
            if "edge_sender_map" in maps:
                row["link_shared_generator_only"] = 1.0
            row.update({f"{name}_fro": float(maps[name].norm()) for name in EDGE_MAP_NAMES if name in maps})
            rows.append(row)
    return rows


def link_eigenvalue_rows(state: Mapping[str, torch.Tensor], cfg: Any) -> List[Dict[str, float]]:
    r"""Long rows (layer, head, k, re, im, abs): eigenvalues of :math:`\hat T_h \hat O_h`, the nonzero spectrum
    of :math:`M_h = \hat O_h \hat T_h`, per head in the order :math:`(-|\lambda|, -\mathrm{Re}, -\mathrm{Im})`;
    similarity invariants of the content links. Layers without content maps are skipped. On a
    ``content_link_edges`` layer these are the eigenvalues of the shared generator's factor (the
    :math:`B = 0` link), and the rows carry ``link_shared_generator_only = 1.0``; the edge links are context
    dependent (see :func:`map_invariants`)."""
    state, _ = unwrap_state(state)
    rows: List[Dict[str, float]] = []
    for layer, maps in relational_maps(state, cfg).items():
        if "T" not in maps or "O" not in maps:
            continue
        H, d, B = _frame_geometry(maps, cfg)
        lam = content_link_eigenvalues(*_ambient_content(maps["T"], maps["O"], d, B))   # (H, d) complex128
        marker = {"link_shared_generator_only": 1.0} if "edge_sender_map" in maps else {}
        for h in range(H):
            for k in range(d):
                z = complex(lam[h, k])
                rows.append({"layer": int(layer), "head": int(h), "k": int(k),
                             "re": z.real, "im": z.imag, "abs": abs(z), **marker})
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
