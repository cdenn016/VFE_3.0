r"""Randomly drawn so_n irrep towers (``irrep_spec="random"``) and the wedge-tensor irrep families.

Pins, on CPU at tiny size: the exterior-power families ``w<p>`` (Lambda^p, 2 <= p <= N/2) and the
self-dual / anti-self-dual ``wp<p>`` / ``wm<p>`` (p = N/2 even): dimensions, skew blocks, the
homomorphism (checked by ``direct_sum_generators``), irreducibility through the commutant dimension
(1 for real type, 2 for the complex-type Lambda^3 of SO(6)), the self-dual split, and the refused
ranks. The sampler: exact block count and total, determinism per seed, variety across seeds, grouped
ascending layout, rank bounds, uniformity over multisets on an enumerable case, and the infeasible
error naming the nearest feasible embed_dim. The config: resolution at construction, a saved config
reloading the drawn spec, independence from ``seed``, and a train step with the
gauge-equivariance residual on both projected routes.
"""
import itertools
import math
import warnings
from collections import Counter

import pytest
import torch

from tests.profiles import test_config, tiny_config
from vfe3.config import config_from_serialized, serialize_config
from vfe3.geometry.generators import generate_son
from vfe3.geometry.irreps import direct_sum_generators, irrep_dim, random_irrep_spec


def _blocks(N: int, spec: list) -> list:
    G, dims = direct_sum_generators(generate_son(N, dtype=torch.float64), algebra="so", irrep_spec=spec)
    out, start = [], 0
    for d in dims:
        out.append(G[:, start:start + d, start:start + d])
        start += d
    return out


def _commutant_dim(rho: torch.Tensor) -> int:
    d = rho.shape[-1]
    eye = torch.eye(d, dtype=torch.float64)
    A = torch.cat([torch.kron(eye, b.contiguous()) - torch.kron(b.T.contiguous(), eye) for b in rho])
    return int((torch.linalg.svdvals(A) < 1e-9).sum())


# ---------------------------------------------------------------- wedge families

@pytest.mark.parametrize("N,label,dim,commutant", [
    (4, "wp2", 3, 1), (4, "wm2", 3, 1), (5, "w2", 10, 1), (6, "w2", 15, 1), (6, "w3", 20, 2),
    (7, "w3", 35, 1), (8, "w3", 56, 1), (8, "wp4", 35, 1), (8, "wm4", 35, 1),
])
def test_wedge_irreps_dims_skew_and_irreducible(N: int, label: str, dim: int, commutant: int) -> None:
    assert irrep_dim(N, algebra="so", label=label) == dim
    (rho,) = _blocks(N, [(label, 1)])
    assert rho.shape == (N * (N - 1) // 2, dim, dim)
    assert float((rho + rho.transpose(-1, -2)).abs().max()) == 0.0
    assert _commutant_dim(rho) == commutant


def test_self_dual_halves_are_inequivalent_and_rebuild_lambda_m() -> None:
    plus, minus = _blocks(4, [("wp2", 1), ("wm2", 1)])
    # Casimirs agree but the halves differ: sum_a rho(L_a) restricted to the so(3) factors.
    cas_p = -(plus @ plus).sum(0)
    cas_m = -(minus @ minus).sum(0)
    torch.testing.assert_close(cas_p, cas_m)
    # Inequivalent: no nonzero intertwiner X with rho_+ X = X rho_- for every generator.
    eye = torch.eye(3, dtype=torch.float64)
    A = torch.cat([torch.kron(eye, p.contiguous()) - torch.kron(m.T.contiguous(), eye) for p, m in zip(plus, minus)])
    assert int((torch.linalg.svdvals(A) < 1e-9).sum()) == 0


@pytest.mark.parametrize("N,label", [(8, "w1"), (8, "w5"), (8, "w4"), (6, "wp3"), (5, "wm2")])
def test_wedge_refusals(N: int, label: str) -> None:
    with pytest.raises(ValueError):
        irrep_dim(N, algebra="so", label=label)


# ---------------------------------------------------------------- sampler

@pytest.mark.parametrize("N,K,H", [(3, 38, 4), (4, 64, 8), (5, 64, 8), (6, 96, 6), (8, 124, 8)])
def test_sampler_hits_count_and_total_deterministically(N: int, K: int, H: int) -> None:
    for seed in range(4):
        spec = random_irrep_spec(N, K, H, seed=seed)
        assert spec == random_irrep_spec(N, K, H, seed=seed)
        assert sum(m for _, m in spec) == H
        assert sum(m * irrep_dim(N, algebra="so", label=label) for label, m in spec) == K
        assert all(m >= 1 for _, m in spec) and len({label for label, _ in spec}) == len(spec)


def test_sampler_varies_and_respects_rank_bounds() -> None:
    draws = {tuple(random_irrep_spec(4, 64, 8, seed=s)) for s in range(12)}
    assert len(draws) > 3
    for s in range(6):
        spec = random_irrep_spec(3, 38, 4, seed=s, max_rank=8)
        assert all(int(label[1:]) <= 8 for label, _ in spec)
    with_trivial = [random_irrep_spec(3, 12, 4, seed=s, min_rank=0) for s in range(20)]
    assert any(label == "l0" for spec in with_trivial for label, _ in spec)
    assert all(label != "l0" for s in range(10) for label, _ in random_irrep_spec(3, 12, 4, seed=s))


def test_sampler_is_uniform_over_multisets() -> None:
    # SO(3), K=15, three heads, ranks >= 1: the odd-dimension multisets {3,3,9}, {3,5,7}, {5,5,5}.
    dims = {f"l{p}": 2 * p + 1 for p in range(1, 8)}
    expected = {tuple(sorted(c)) for c in itertools.combinations_with_replacement(dims.values(), 3) if sum(c) == 15}
    assert expected == {(3, 3, 9), (3, 5, 7), (5, 5, 5)}
    counts = Counter(tuple(sorted(dims[label] for label, m in random_irrep_spec(3, 15, 3, seed=s) for _ in range(m)))
                     for s in range(3000))
    assert set(counts) == expected
    assert all(abs(n - 1000) < 150 for n in counts.values()), counts


def test_sampler_layout_is_grouped_and_ascending() -> None:
    spec = random_irrep_spec(4, 64, 8, seed=0)
    keys = [(label.rstrip("0123456789"), int(label[len(label.rstrip("0123456789")):])) for label, _ in spec]
    assert keys == sorted(keys)


def test_infeasible_draw_names_the_nearest_feasible_embed_dim() -> None:
    with pytest.raises(ValueError, match="Nearest feasible embed_dim for n_heads=8: 124 below, 131 above"):
        random_irrep_spec(8, 128, 8)
    with pytest.raises(ValueError, match="37 below, 39 above"):        # SO(3): odd blocks, parity
        random_irrep_spec(3, 38, 5)
    assert random_irrep_spec(8, 64, 8, seed=5) == [("l1", 8)]           # K / H = smallest block: forced


# ---------------------------------------------------------------- config and model

def test_config_resolves_reloads_and_ignores_the_training_seed() -> None:
    base = dict(gauge_group="so_n", group_n=4, embed_dim=64, n_heads=8, irrep_spec="random")
    cfg = test_config(**base, irrep_spec_seed=1)
    assert isinstance(cfg.irrep_spec, list) and len(cfg.head_dims) == 8 and sum(cfg.head_dims) == 64
    assert cfg.irrep_spec == random_irrep_spec(4, 64, 8, seed=1)
    assert config_from_serialized(serialize_config(cfg), source="test").irrep_spec == cfg.irrep_spec
    assert test_config(**base, irrep_spec_seed=1, seed=99).irrep_spec == cfg.irrep_spec
    assert {tuple(test_config(**base, irrep_spec_seed=s).irrep_spec) for s in range(8)} != {tuple(cfg.irrep_spec)}
    with pytest.raises(ValueError, match="'random'"):
        test_config(**{**base, "irrep_spec": "rand"})
    with pytest.raises(ValueError, match="Nearest feasible"):
        test_config(**{**base, "group_n": 8, "embed_dim": 128})
    manual = test_config(gauge_group="so_n", group_n=4, embed_dim=10, n_heads=3,
                         irrep_spec=[("l1", 1), ("wp2", 1), ("wm2", 1)])
    assert manual.head_dims == [4, 3, 3]
    assert "irrep_spec_seed" in test_config().inactive_fields()


@pytest.mark.parametrize("route", ["predictive", "gradient"])
def test_random_wedge_tower_trains_and_is_equivariant(route: str) -> None:
    from vfe3.model.model import VFEModel
    from vfe3.train import build_optimizer, train_step
    from vfe3.viz import extract
    extra = (dict(attention_mode="projected_gradient", predictive_covariance="passthrough",
                  attention_residual=False) if route == "gradient" else {})
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(tiny_config(gauge_group="so_n", group_n=4, embed_dim=10, n_heads=3,
                                     irrep_spec=[("l1", 1), ("wp2", 1), ("wm2", 1)], **extra))
    with torch.no_grad():
        model.prior_bank.phi_embed.normal_(mean=0.0, std=0.3)
    residuals = extract.projected_gauge_equivariance(
        model.eval(), torch.tensor([[1, 4, 2]]), n_draws=2)["residuals"]
    assert residuals and all(max(v) < 1e-10 for v in residuals.values()), residuals
    model.train()
    optimizer = build_optimizer(model, model.cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    assert math.isfinite(train_step(model, optimizer, scheduler, torch.tensor([[1, 4, 2]]),
                                    torch.tensor([[4, 2, 3]])))
