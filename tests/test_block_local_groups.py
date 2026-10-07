r"""Plan A of the gauge-group buildout (gauge-group-buildouts-2026-10-05.md): ``block_so`` and ``block_co``.

``block_so`` is SO(d)^H, an independent rotation per head; ``block_co`` is CO(d)^H = (R+ x SO(d))^H,
a rotation times a positive per-head gain. Pins, on CPU at tiny size: the generator structure and
bracket closure; exponentials landing in the group (U^T U = I, det 1; U^T U = lambda^2 I, det > 0);
admissibility for the full Gaussian with the diagonal family as the negative control; config
acceptance and refusals; RoPE staying inside the group; and a projected train step, the
gauge-equivariance residual and a float64 finite-difference phi gradient on both projected routes.
"""
import math
import warnings

import pytest
import torch

from tests.profiles import test_config, tiny_config
from vfe3.geometry.groups import check_admissible, get_group, group_traits
from vfe3.model.model import VFEModel

GROUPS = ("block_so", "block_co")
N_GEN = {"block_so": lambda d: d * (d - 1) // 2, "block_co": lambda d: d * (d - 1) // 2 + 1}
ROUTES = {
    "predictive": {},
    "predictive_full": dict(family="gaussian_full", decode_mode="full_chunked"),
    "gradient": dict(attention_mode="projected_gradient", predictive_covariance="passthrough",
                     attention_residual=False),
}


def _model(name: str, dtype: torch.dtype = torch.float32, **overrides: object) -> VFEModel:
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(tiny_config(gauge_group=name, embed_dim=6, n_heads=2, **overrides)).to(dtype)
    with torch.no_grad():
        model.prior_bank.phi_embed.normal_(mean=0.0, std=0.3)
    return model


@pytest.mark.parametrize("name", GROUPS)
def test_structure_and_bracket_closure(name: str) -> None:
    group = get_group(name)(6, 2, dtype=torch.float64)
    G = group.generators
    m = N_GEN[name](3)
    assert G.shape == (2 * m, 6, 6) and group.irrep_dims == [3, 3] and group.tying == "per_head"
    assert group.skew_symmetric == (name == "block_so") and group.pullback_capability is None
    torch.testing.assert_close(torch.einsum("aij,bij->ab", G, G), torch.eye(2 * m, dtype=torch.float64))
    for a in range(2 * m):                                   # each generator lives on one head block
        assert (float(G[a, :3, :3].abs().sum()) > 0) != (float(G[a, 3:, 3:].abs().sum()) > 0)
        assert float(G[a, :3, 3:].abs().max()) == 0.0 and float(G[a, 3:, :3].abs().max()) == 0.0
    flat = G.reshape(2 * m, -1)
    for a in range(2 * m):
        for b in range(2 * m):
            c = (G[a] @ G[b] - G[b] @ G[a]).reshape(-1)
            assert float((c - flat.T @ (flat @ c)).abs().max()) < 1e-12
    traits = group_traits(name)
    assert traits.algebra is None and traits.tyings == ("per_head",) and traits.contains_pair_rotations


@pytest.mark.parametrize("name", GROUPS)
def test_exponentials_land_in_the_group(name: str) -> None:
    group = get_group(name)(8, 2, dtype=torch.float64)
    phi = torch.randn(6, group.generator_shape[0], generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    U = torch.linalg.matrix_exp(torch.einsum("na,aij->nij", phi, group.generators))
    for h in range(2):
        block = U[:, 4 * h:4 * h + 4, 4 * h:4 * h + 4]
        gram = block.transpose(-1, -2) @ block
        scale = torch.diagonal(gram, dim1=-2, dim2=-1).mean(-1)
        torch.testing.assert_close(gram, scale[:, None, None] * torch.eye(4, dtype=torch.float64))
        det = torch.linalg.det(block)
        if name == "block_so":
            torch.testing.assert_close(scale, torch.ones_like(scale))
            torch.testing.assert_close(det, torch.ones_like(det))
        else:
            assert bool((det > 0).all()) and float((scale - 1.0).abs().max()) > 1e-3   # a genuine gain
    assert float(U[:, :4, 4:].abs().max()) == 0.0


@pytest.mark.parametrize("name", GROUPS)
def test_admissibility_with_the_diagonal_family_as_negative_control(name: str) -> None:
    group = get_group(name)(6, 2, dtype=torch.float64)
    assert check_admissible(group, "gaussian_full")
    assert not check_admissible(group, "gaussian_diagonal")
    assert group.invariant_for("gaussian_full") and not group.invariant_for("gaussian_diagonal")


def test_config_accepts_and_refuses() -> None:
    for name in GROUPS:
        cfg = test_config(gauge_group=name, embed_dim=6, n_heads=2)
        assert cfg.gauge_tying == "per_head" and cfg.head_dims == [3, 3]
        with pytest.raises(ValueError, match="gauge_tying"):
            test_config(gauge_group=name, embed_dim=6, n_heads=2, gauge_tying="shared")
        with pytest.raises(ValueError, match="divisible"):
            test_config(gauge_group=name, embed_dim=7, n_heads=2)
        with pytest.raises(ValueError, match="irrep-tower"):
            test_config(gauge_group=name, embed_dim=6, n_heads=2, group_n=3)
        with pytest.raises(ValueError, match="irrep-labeled tower"):
            test_config(gauge_group=name, embed_dim=6, n_heads=2, use_cg_coupling=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error")                    # RoPE stays inside the group: no warning
            test_config(gauge_group=name, embed_dim=6, n_heads=2, pos_rotation="rope")
    with pytest.raises(ValueError, match="head blocks of size >= 2"):
        test_config(gauge_group="block_so", embed_dim=4, n_heads=4)
    assert test_config(gauge_group="block_co", embed_dim=4, n_heads=4).head_dims == [1, 1, 1, 1]


@pytest.mark.parametrize("name", GROUPS)
@pytest.mark.parametrize("route", sorted(ROUTES))
def test_train_step_and_gauge_equivariance(name: str, route: str) -> None:
    from vfe3.train import build_optimizer, train_step
    from vfe3.viz import extract
    model = _model(name, **ROUTES[route])
    assert model.prior_bank.phi_embed.shape[-1] == 2 * N_GEN[name](3)
    residuals = extract.projected_gauge_equivariance(
        model.eval(), torch.tensor([[1, 4, 2]]), n_draws=2)["residuals"]
    assert residuals and all(max(v) < 1e-10 for v in residuals.values()), residuals
    model.train()
    optimizer = build_optimizer(model, model.cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    phi0 = model.prior_bank.phi_embed.detach().clone()
    loss = train_step(model, optimizer, scheduler, torch.tensor([[1, 4, 2]]), torch.tensor([[4, 2, 3]]))
    assert math.isfinite(loss)
    assert float((model.prior_bank.phi_embed.detach() - phi0).abs().max()) > 0.0


@pytest.mark.parametrize("name", GROUPS)
def test_rope_train_step(name: str) -> None:
    from vfe3.train import build_optimizer, train_step
    model = _model(name, pos_rotation="rope", rope_on_value=False, n_layers=2)
    optimizer = build_optimizer(model, model.cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    assert math.isfinite(train_step(model, optimizer, scheduler, torch.tensor([[1, 4, 2]]),
                                    torch.tensor([[4, 2, 3]])))


@pytest.mark.parametrize("name", GROUPS)
@pytest.mark.parametrize("route", ["predictive", "gradient"])
def test_phi_gradient_matches_finite_difference(name: str, route: str) -> None:
    model = _model(name, torch.float64, **ROUTES[route]).eval()
    tokens = torch.tensor([[1, 4, 2]])
    phi = model.prior_bank.phi_embed
    w_mu = torch.randn(1, 3, 6, generator=torch.Generator().manual_seed(3), dtype=torch.float64)

    def objective() -> torch.Tensor:
        belief, _ = model.forward_beliefs(tokens)
        w_sigma = torch.randn(belief.sigma.shape, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
        return (w_mu * belief.mu).sum() + (w_sigma * belief.sigma).sum()

    objective().backward()
    grad = phi.grad.clone()
    h = 1e-5
    last = phi.shape[-1] - 1                                  # block_co: the second head's gain coordinate
    for row, col in [(1, 0), (4, 1), (2, last)]:
        with torch.no_grad():
            phi[row, col] += h
            plus = float(objective())
            phi[row, col] -= 2 * h
            minus = float(objective())
            phi[row, col] += h
        fd = (plus - minus) / (2 * h)
        assert abs(fd - float(grad[row, col])) <= 1e-7 * max(1.0, abs(fd)), (row, col, fd, grad[row, col])
