r"""Triangle holonomy of the head-sum link field of a trained run (link_curvature note, Research vault).

For the projected routes the reference-coordinate edge operator is
:math:`\hat O B_{ij}\hat T` with :math:`B_{ij}=\bigoplus_h\beta^h_{ij}I_d`. With tiled heads
(:math:`Hd=K`) the triangle holonomy :math:`F_{ijk}=L_{ik}^{-1}L_{ij}L_{jk}` on a strict triangle
:math:`k<j<i` is conjugate to

.. math::
    R_{ijk}G,\qquad G=\hat T\hat O,\qquad R_{ijk}=\bigoplus_h\rho^h_{ijk}I_d,\qquad
    \rho^h_{ijk}=\frac{\beta^h_{ij}\beta^h_{jk}}{\beta^h_{ik}},

so its spectrum is a gauge invariant and the frames drop out. :math:`G` is the routing-independent
factor (the constant-link curvature, readable from every map snapshot) and :math:`R` the routing factor
(flat exactly for :math:`\beta_{ij}=c_i/c_j`). The curvature size of a matrix with eigenvalues
:math:`\lambda_k` is :math:`\kappa=(\tfrac1K\sum_k|\mathrm{Log}\,\lambda_k|^2)^{1/2}` (principal complex
logarithm), zero exactly when every eigenvalue is 1 (for a diagonalizable matrix, the identity).

Two measurements:

* :func:`checkpoint_holonomy`: on evenly spaced validation windows of one full checkpoint, the exact
  path-weighted moments of :math:`\log\rho^h` per (layer, head) for the router and its prior
  (:func:`vfe3.metrics.routing_triangle_defect_sums`, the same quantity as the training-time
  ``diagnostics/routing_holonomy.csv``), and :math:`\kappa` of :math:`R_{ijk}G`, :math:`G` and
  :math:`R_{ijk}` on triangles sampled with probability proportional to the head-mean path weight
  :math:`\bar\beta_{ij}\bar\beta_{jk}`.
* :func:`snapshot_holonomy`: the spectrum invariants of :math:`G` per layer at every map snapshot.

Run as ``python -m vfe3.posthoc.holonomy RUN_DIR [--checkpoints ...]``; writes JSON.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

from vfe3.metrics import routing_triangle_defect_sums, triangle_holonomy_eigenvalues
from vfe3.posthoc.maps import _ambient_content, _frame_geometry, load_map_state, relational_maps, snapshot_files

QUANTILES = (0.1, 0.5, 0.9)


def curvature_size(lam: torch.Tensor, scale_free: bool = False) -> torch.Tensor:   # (..., K) complex -> (...,)
    r""":math:`\kappa=(\tfrac1K\sum_k|\mathrm{Log}\,\lambda_k|^2)^{1/2}`; infinite for a singular matrix.

    ``scale_free`` first removes the central scale :math:`\tfrac1K\log|\det|` from every
    :math:`\log|\lambda_k|`: the curvature of the :math:`\mathrm{SL}^{\pm}(K)` part, zero for a scalar
    holonomy :math:`cI` with :math:`c>0`.
    """
    log_abs = torch.log(lam.abs())
    if scale_free:
        log_abs = log_abs - log_abs.mean(dim=-1, keepdim=True)
    return torch.sqrt((log_abs.square() + torch.angle(lam).square()).mean(dim=-1))


def layer_content_maps(state: Mapping[str, torch.Tensor], cfg: Any) -> Dict[int, tuple]:
    r"""``{layer: (T_hat (H, d, K), O_hat (H, K, d))}`` float64, for layers with content maps."""
    out: Dict[int, tuple] = {}
    for layer, maps in relational_maps(state).items():
        if "T" in maps and "O" in maps:
            _, d, B = _frame_geometry(maps, cfg)
            out[layer] = _ambient_content(maps["T"], maps["O"], d, B)
    return out


def g_invariants(T_hat: torch.Tensor, O_hat: torch.Tensor) -> Dict[str, float]:
    r"""Spectrum invariants of :math:`G=\hat T\hat O`: :math:`\kappa`, min and max modulus, mean log-modulus,
    largest :math:`|\arg\lambda|`, and the count of eigenvalues on the negative real axis."""
    lam = triangle_holonomy_eigenvalues(T_hat, O_hat, torch.ones(1, T_hat.shape[0]))[0]   # (K,)
    neg = ((lam.real < 0) & (lam.imag.abs() < 1e-12 * lam.abs().clamp_min(1e-300))).sum()
    return {"G_kappa": float(curvature_size(lam)), "G_kappa_scale_free": float(curvature_size(lam, True)),
            "G_min_abs": float(lam.abs().min()),
            "G_max_abs": float(lam.abs().max()), "G_mean_log_abs": float(torch.log(lam.abs()).mean()),
            "G_max_abs_arg": float(torch.angle(lam).abs().max()), "G_negative_real": int(neg)}


def _quantiles(values: torch.Tensor) -> Dict[str, float]:
    v = values[torch.isfinite(values)].double()
    if not v.numel():
        return {f"q{int(q * 100)}": float("nan") for q in QUANTILES}
    qs = torch.quantile(v, torch.tensor(QUANTILES, dtype=torch.float64))
    return {f"q{int(q * 100)}": float(x) for q, x in zip(QUANTILES, qs)}


class RoutingHolonomyObserver:
    r"""``nll_pass`` observer: defect moments per (layer, head) and path-sampled triangle defects."""

    def __init__(self, n_triangles: int = 256, seed: int = 0) -> None:
        self.n_triangles = int(n_triangles)
        self.generator = torch.Generator().manual_seed(int(seed))
        self.sums: Dict[int, torch.Tensor] = {}        # layer -> (H, 6) learned and prior moments
        self.rows: Dict[int, float] = {}
        self.rho: Dict[int, List[torch.Tensor]] = {}   # layer -> [(S, H) float64]

    def update(self, diagnostic: Mapping[str, Any], valid: torch.Tensor, meta: Mapping[str, Any]) -> None:
        log_prior = diagnostic.get("log_prior")
        for layer, capture in enumerate(diagnostic.get("layer_conditional_captures") or []):
            beta = capture["beta_pre"].double()
            if beta.dim() == 3:
                beta = beta.unsqueeze(1)                                       # (B, H, N, N)
            Bn, H, N, _ = beta.shape
            if isinstance(log_prior, torch.Tensor):
                logits = log_prior.double()
                while logits.dim() < 4:
                    logits = logits.unsqueeze(0)
                support = torch.isfinite(logits).expand_as(beta)
                prior = torch.softmax(logits, dim=-1).expand_as(beta)
            else:
                support = beta > 0.0
                prior = support.double()
            prior = torch.where(support, prior, torch.zeros_like(prior))
            prior = prior / prior.sum(dim=-1, keepdim=True).clamp_min(1e-30)
            moments = torch.cat([routing_triangle_defect_sums(beta, support),
                                 routing_triangle_defect_sums(prior, support)], dim=-1)   # (H, 6)
            self.sums[layer] = self.sums.get(layer, 0) + moments
            self.rows[layer] = self.rows.get(layer, 0.0) + float(Bn * N)
            self.rho.setdefault(layer, []).append(self._sample(beta.cpu(), support.cpu()))

    def _sample(self, beta: torch.Tensor, support: torch.Tensor) -> torch.Tensor:   # (S, H)
        Bn, H, N, _ = beta.shape
        strict = torch.ones(N, N, dtype=torch.bool).tril(-1)
        edge = (support & strict & (beta > 0)).all(dim=1)                     # (B, N, N) every head
        mean = torch.where(edge, beta.mean(dim=1), torch.zeros(()))          # (B, N, N) head-mean strict
        out = []
        for b in range(Bn):
            m = mean[b]
            # P(i, j) ~ m_ij * sum_{k: edge_ik} m_jk, then k ~ m_jk restricted to edge_ik.
            onward = m @ edge[b].double().T                                    # (N, N): [j, i] -> sum_k m_jk e_ik
            pair = m * onward.T                                                # (i, j)
            total = float(pair.sum())
            if total <= 0.0:
                continue
            idx = torch.multinomial(pair.reshape(-1) / total, self.n_triangles, replacement=True,
                                    generator=self.generator)
            i, j = idx // N, idx % N
            wk = m[j] * edge[b][i].double()                                    # (S, N)
            k = torch.multinomial(wk / wk.sum(dim=-1, keepdim=True), 1, generator=self.generator)[:, 0]
            bb = beta[b]                                                       # (H, N, N)
            out.append((bb[:, i, j] * bb[:, j, k] / bb[:, i, k]).T)            # (S, H)
        return torch.cat(out) if out else torch.zeros(0, H, dtype=torch.float64)

    def finalize(self) -> None:
        return None


def checkpoint_holonomy(
    run_dir:    "str | Path",
    checkpoint: str,

    *,
    n_windows:   int = 8,
    batch_size:  int = 4,
    n_triangles: int = 256,
    seed:        int = 0,
    device:      str = "cpu",
) -> Dict[str, Any]:
    r"""Defect moments and triangle-holonomy curvature of one checkpoint on validation windows."""
    from vfe3.posthoc import evaluate as ev
    from vfe3.posthoc import streams as st

    run_dir = Path(run_dir)
    loaded = ev.load_model(run_dir, checkpoint, device=ev.resolve_device(device))
    info = st.load_run_info(run_dir, checkpoint)
    tokens, record = st.resolve_stream(info, "validation")
    dataset = st.eval_windows(tokens, info.seq_len, info.eval_stride)
    usable = len(dataset) - 1 if len(dataset) > 1 else len(dataset)              # skip the padded tail
    indices = st.evenly_spaced(usable, n_windows)
    observer = RoutingHolonomyObserver(n_triangles=n_triangles, seed=seed)
    result = ev.nll_pass(loaded.model, dataset, indices, batch_size=batch_size,
                         device=ev.resolve_device(device), observers=[observer])
    maps = layer_content_maps(loaded.model.state_dict(), loaded.cfg)
    layers: List[Dict[str, Any]] = []
    for layer in sorted(observer.sums):
        sums, n_rows = observer.sums[layer], observer.rows[layer]
        heads = []
        for h in range(sums.shape[0]):
            mass, s1, s2, p_mass, p1, p2 = sums[h].tolist()
            heads.append({
                "head": h, "path_mass": mass / n_rows,
                "log_defect_mean": s1 / mass if mass > 0 else float("nan"),
                "log_defect_rms": math.sqrt(s2 / mass) if mass > 0 else float("nan"),
                "prior_log_defect_mean": p1 / p_mass if p_mass > 0 else float("nan"),
                "prior_log_defect_rms": math.sqrt(p2 / p_mass) if p_mass > 0 else float("nan"),
            })
        entry: Dict[str, Any] = {"layer": layer, "heads": heads}
        rho = torch.cat(observer.rho[layer]) if observer.rho[layer] else torch.zeros(0, 1)
        entry["n_triangles"] = int(rho.shape[0])
        entry["R_kappa"] = _quantiles(torch.sqrt(torch.log(rho).square().mean(dim=-1)))
        if layer in maps:
            T_hat, O_hat = maps[layer]
            entry.update(g_invariants(T_hat, O_hat))
            if rho.shape[0] and T_hat.shape[0] * T_hat.shape[1] == T_hat.shape[-1]:
                lam = triangle_holonomy_eigenvalues(T_hat, O_hat, rho)        # (S, K)
                entry["F_kappa"] = _quantiles(curvature_size(lam))
                entry["F_kappa_scale_free"] = _quantiles(curvature_size(lam, True))
                entry["F_mean_log_abs"] = _quantiles(torch.log(lam.abs()).mean(dim=-1))
        layers.append(entry)
    return {"checkpoint": checkpoint, "step": _step_of(checkpoint), "n_windows": int(len(indices)),
            "seq_len": info.seq_len, "scored_targets": result.n, "stream_status": record.get("status"),
            "notes": loaded.notes, "layers": layers}


def snapshot_holonomy(run_dir: "str | Path", cfg: Any) -> List[Dict[str, Any]]:
    r""":func:`g_invariants` per layer at every ``map_snapshots/step_*.pt`` of the run."""
    rows: List[Dict[str, Any]] = []
    directory = Path(run_dir) / "map_snapshots"
    if not directory.is_dir():
        return rows
    for step, path in snapshot_files(directory):
        state, _ = load_map_state(path)
        for layer, (T_hat, O_hat) in layer_content_maps(state, cfg).items():
            rows.append({"step": int(step), "layer": int(layer), **g_invariants(T_hat, O_hat)})
    return rows


def _step_of(checkpoint: str) -> Optional[int]:
    stem = Path(checkpoint).stem
    return int(stem.split("_")[-1]) if stem.startswith("step_") and stem.split("_")[-1].isdigit() else None


def default_checkpoints(run_dir: Path) -> List[str]:
    found = sorted((p for p in (run_dir / "checkpoints").glob("step_*.pt")),
                   key=lambda p: _step_of(p.name) or 0)
    if found:
        return [str(p.relative_to(run_dir)).replace("\\", "/") for p in found]
    return ["final_model.pt"]


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--checkpoints", nargs="*", default=None)
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--triangles", type=int, default=256, help="sampled triangles per window and layer")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    from vfe3.posthoc import streams as st

    checkpoints = args.checkpoints or default_checkpoints(args.run_dir)
    report: Dict[str, Any] = {"run_dir": str(args.run_dir), "checkpoints": []}
    for checkpoint in checkpoints:
        print(f"[holonomy] {checkpoint}", flush=True)
        report["checkpoints"].append(checkpoint_holonomy(
            args.run_dir, checkpoint, n_windows=args.windows, batch_size=args.batch_size,
            n_triangles=args.triangles, seed=args.seed, device=args.device))
    cfg = st.load_run_info(args.run_dir, checkpoints[-1]).cfg
    report["snapshots"] = snapshot_holonomy(args.run_dir, cfg)
    out = args.out or (args.run_dir / "posthoc" / "routing_holonomy.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"[holonomy] wrote {out}")
    return report


if __name__ == "__main__":
    main()
