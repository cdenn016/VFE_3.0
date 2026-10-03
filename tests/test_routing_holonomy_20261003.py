"""Triangle holonomy of the head-sum link field (link_curvature note, 2026-10-03).

Tiny CPU tensors and models only (K=4, two heads, N <= 6). Covers the exact path-weighted defect
moments against brute force, zero defect for flat routing beta_ij = c_i / c_j, the conjugacy
F_ijk ~ R_ijk G against the direct product of edge operators, the training-time
diagnostics/routing_holonomy.csv, and the post-hoc observer's sampled triangles.
"""
from tests.profiles import test_config
import csv
import logging
import math
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch
from torch.utils.data import DataLoader

import vfe3.run_artifacts as ra
import vfe3.train as T
from vfe3.data.datasets import TokenWindows
from vfe3.metrics import routing_triangle_defect_sums, triangle_holonomy_eigenvalues
from vfe3.model.model import VFEModel
from vfe3.posthoc.holonomy import RoutingHolonomyObserver, curvature_size

B, N, V = 2, 4, 7


def _causal_beta(Bn: int, H: int, n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(Bn, H, n, n, generator=g, dtype=torch.float64)
    mask = torch.ones(n, n, dtype=torch.bool).tril()
    return torch.softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)


def _brute(beta: torch.Tensor) -> torch.Tensor:                          # (H, 3)
    Bn, H, n, _ = beta.shape
    out = torch.zeros(H, 3, dtype=torch.float64)
    for b in range(Bn):
        for h in range(H):
            for i in range(n):
                for j in range(i):
                    for k in range(j):
                        w = beta[b, h, i, j] * beta[b, h, j, k]
                        x = torch.log(w / beta[b, h, i, k])
                        out[h] += torch.stack([w, w * x, w * x * x])
    return out


def test_defect_sums_match_brute_force() -> None:
    beta = _causal_beta(2, 3, 6, seed=1)
    torch.testing.assert_close(routing_triangle_defect_sums(beta), _brute(beta), rtol=1e-10, atol=1e-12)
    support = torch.ones(6, 6, dtype=torch.bool).tril()
    torch.testing.assert_close(routing_triangle_defect_sums(beta, support), _brute(beta),
                               rtol=1e-10, atol=1e-12)


def test_flat_routing_has_zero_defect_and_uniform_does_not() -> None:
    c = torch.tensor([1.0, 0.5, 0.2, 0.05, 0.01], dtype=torch.float64)
    n = c.numel()
    beta = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n):
        for j in range(i):
            beta[i, j] = c[i] / c[j] if j > 0 else c[i]
        beta[i, i] = 1.0 - beta[i, :i].sum()
    assert (beta.diagonal() > 0).all()
    moments = routing_triangle_defect_sums(beta[None, None])
    assert moments[0, 0] > 0
    assert abs(float(moments[0, 1])) < 1e-12 and float(moments[0, 2]) < 1e-12        # expansion roundoff
    uniform = torch.ones(n, n, dtype=torch.float64).tril()
    uniform = uniform / uniform.sum(dim=-1, keepdim=True)
    assert float(routing_triangle_defect_sums(uniform[None, None])[0, 2]) > 1e-3


def test_holonomy_is_conjugate_to_routing_times_transfer() -> None:
    g = torch.Generator().manual_seed(3)
    H, d = 2, 2
    K = H * d
    T_hat = torch.randn(H, d, K, generator=g, dtype=torch.float64)
    O_hat = torch.randn(H, K, d, generator=g, dtype=torch.float64)
    beta = _causal_beta(1, H, 4, seed=4)[0]                                  # (H, N, N)

    def edge(i: int, j: int) -> torch.Tensor:                                # sum_h beta^h O_h T_h
        return sum(beta[h, i, j] * O_hat[h] @ T_hat[h] for h in range(H))

    i, j, k = 3, 2, 0
    F = torch.linalg.solve(edge(i, k), edge(i, j) @ edge(j, k))
    rho = (beta[:, i, j] * beta[:, j, k] / beta[:, i, k])[None]              # (1, H)
    lam = triangle_holonomy_eigenvalues(T_hat, O_hat, rho)[0]
    direct = torch.linalg.eigvals(F)
    key = lambda z: sorted((round(v.real, 8), round(v.imag, 8)) for v in z.tolist())
    assert key(lam) == key(direct)
    assert float(curvature_size(torch.ones(K, dtype=torch.complex128))) == 0.0


def _cfg(**overrides: Any) -> Any:
    values: Dict[str, Any] = dict(
        vocab_size=V, embed_dim=4, n_heads=2, batch_size=B, max_seq_len=N, eval_stride=2,
        n_layers=2, n_e_steps=1, warmup_steps=0, max_steps=6, log_interval=0, eval_interval=0,
        checkpoint_interval=0, skip_belief_sigma_update=True,
        generate_embedding_figures=False, pos_phi="learned",
        use_block_mlp=True, block_mlp_expansion=1, beta_attention_prior="causal_alibi",
        family="gaussian_diagonal", gauge_group="block_glk",
        attention_mode="projected_predictive", attention_residual=True,
        norm_type_block="layernorm", layernorm_affine=True, decode_mode="linear")
    values.update(overrides)
    return test_config(**values)


def _rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(autouse=True)
def _no_heatmap_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(T, "_save_eval_attention_maps", lambda *args, **kwargs: None)
    monkeypatch.setattr(ra, "_run_figures_isolated", lambda *args, **kwargs: True)
    torch.set_num_threads(1)


def test_training_writes_routing_holonomy_csv(tmp_path: Path) -> None:
    cfg = _cfg()
    torch.manual_seed(5)
    model = VFEModel(cfg)
    train_ds = TokenWindows(torch.arange(41) % V, N)
    val_ds = TokenWindows((torch.arange(23) * 3) % V, N, pad_final=True, stride=cfg.eval_stride)
    loaders = (DataLoader(train_ds, batch_size=B, shuffle=True),
               DataLoader(val_ds, batch_size=2, shuffle=False))
    run_dir = tmp_path / "run"
    artifacts = ra.RunArtifacts(run_dir, cfg, model, save_model_artifacts=False)
    torch.manual_seed(11)
    T.train(model, loaders[0], cfg, n_steps=cfg.max_steps, grad_clip=1.0, log_interval=0,
            eval_interval=3, val_loader=loaders[1], device=torch.device("cpu"),
            logger=logging.getLogger("holonomy-test"), artifacts=artifacts, generate_samples=False)
    rows = _rows(run_dir / "diagnostics" / "routing_holonomy.csv")
    assert list(rows[0]) == list(T.ROUTING_HOLONOMY_FIELDS)
    steps = {int(r["step"]) for r in rows}
    assert len(rows) == len(steps) * cfg.n_layers * cfg.n_heads
    for r in rows:
        assert float(r["path_mass"]) > 0.0
        assert float(r["log_defect_rms"]) >= abs(float(r["log_defect_mean"])) - 1e-12
        assert math.isfinite(float(r["prior_log_defect_rms"]))


def test_observer_samples_exact_defects() -> None:
    beta = _causal_beta(2, 2, 5, seed=7).float()
    observer = RoutingHolonomyObserver(n_triangles=64, seed=0)
    observer.update({"layer_conditional_captures": [{"beta_pre": beta}], "log_prior": None},
                    torch.ones(2, 5, dtype=torch.bool), {})
    torch.testing.assert_close(observer.sums[0][:, :3], _brute(beta.double()), rtol=1e-6, atol=1e-9)
    rho = observer.rho[0][0]
    assert rho.shape == (128, 2) and torch.isfinite(rho).all() and (rho > 0).all()
