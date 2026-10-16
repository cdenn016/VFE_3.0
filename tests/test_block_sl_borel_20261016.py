r"""Plan C of the gauge-group buildout (gauge-group-buildouts-2026-10-05.md): ``block_sl``, ``block_borel`` and
``block_unipotent``.

``block_sl`` is SL(d)^H (det 1 per head), ``block_borel`` B+(d)^H (upper triangular, positive diagonal) and
``block_unipotent`` N+(d)^H (upper triangular, unit diagonal). Pins, on CPU at tiny size: the local bases and
bracket closure; exponentials landing in the group; the trace projection being the identity on the det-1 groups;
admissibility with the diagonal family as the negative control; config acceptance, refusals and the RoPE warning
for the groups without rotations; the frame condition bound on ``block_sl``; and a projected train step, the
gauge-equivariance residual with a zero log-det shift on the det-1 groups, a pullback-group train step and a
float64 finite-difference phi gradient.
"""
import math
import warnings

import pytest
import torch

from tests.profiles import test_config, tiny_config
from vfe3.config import ConfigNotice
from vfe3.geometry.generators import local_basis_borel, local_basis_sl, local_basis_unipotent
from vfe3.geometry.groups import check_admissible, get_group, group_traits
from vfe3.geometry.lie_ops import project_phi_to_slk
from vfe3.model.model import VFEModel

F64 = torch.float64
GROUPS = ("block_sl", "block_borel", "block_unipotent")
N_GEN = {"block_sl": lambda d: d * d - 1, "block_borel": lambda d: d * (d + 1) // 2,
         "block_unipotent": lambda d: d * (d - 1) // 2}
DET_ONE = ("block_sl", "block_unipotent")
ROUTES = {
    "predictive": {},
    "predictive_full": dict(family="gaussian_full", decode_mode="full_chunked"),
    "gradient": dict(attention_mode="projected_gradient", predictive_covariance="passthrough",
                     attention_residual=False),
}


def _quiet_config(**overrides: object):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tiny_config(**overrides)


def _model(name: str, dtype: torch.dtype = torch.float32, **overrides: object) -> VFEModel:
    torch.manual_seed(0)
    overrides = {"rope_mode": "none", **overrides}
    model = VFEModel(_quiet_config(gauge_group=name, embed_dim=6, n_heads=2, **overrides)).to(dtype)
    with torch.no_grad():
        model.prior_bank.phi_embed.normal_(mean=0.0, std=0.3)
    return model


def test_local_bases() -> None:
    sl, borel, unipotent = local_basis_sl(4), local_basis_borel(4), local_basis_unipotent(4)
    assert sl.shape == (15, 4, 4) and borel.shape == (10, 4, 4) and unipotent.shape == (6, 4, 4)
    for basis in (sl, borel, unipotent):
        torch.testing.assert_close(torch.einsum("aij,bij->ab", basis, basis), torch.eye(basis.shape[0], dtype=F64))
    assert float(sl.diagonal(dim1=-2, dim2=-1).sum(-1).abs().max()) < 1e-15            # traceless
    assert float(torch.tril(borel, diagonal=-1).abs().max()) == 0.0                    # upper triangular
    assert float(torch.tril(unipotent).abs().max()) == 0.0                             # strictly upper
    for i in range(4):
        for j in range(i + 1, 4):                                                       # n+ is the off-diagonal part of b+
            E = torch.zeros(4, 4, dtype=F64)
            E[i, j] = 1.0
            assert any(torch.equal(E, u) for u in unipotent)
    with pytest.raises(ValueError):
        local_basis_unipotent(1)


@pytest.mark.parametrize("name", GROUPS)
def test_structure_and_bracket_closure(name: str) -> None:
    group = get_group(name)(6, 2, dtype=F64)
    G = group.generators
    m = N_GEN[name](3)
    assert G.shape == (2 * m, 6, 6) and group.irrep_dims == [3, 3] and group.tying == "per_head"
    assert not group.skew_symmetric and group.pullback_capability == "pullback_per_block"
    torch.testing.assert_close(torch.einsum("aij,bij->ab", G, G), torch.eye(2 * m, dtype=F64))
    for a in range(2 * m):                                   # each generator lives on one head block
        assert (float(G[a, :3, :3].abs().sum()) > 0) != (float(G[a, 3:, 3:].abs().sum()) > 0)
        assert float(G[a, :3, 3:].abs().max()) == 0.0 and float(G[a, 3:, :3].abs().max()) == 0.0
    flat = G.reshape(2 * m, -1)
    for a in range(2 * m):
        for b in range(2 * m):
            c = (G[a] @ G[b] - G[b] @ G[a]).reshape(-1)
            assert float((c - flat.T @ (flat @ c)).abs().max()) < 1e-12
    traits = group_traits(name)
    assert traits.algebra is None and traits.tyings == ("per_head",)
    assert traits.contains_pair_rotations == (name == "block_sl")


@pytest.mark.parametrize("name", GROUPS)
def test_exponentials_land_in_the_group(name: str) -> None:
    group = get_group(name)(8, 2, dtype=F64)
    phi = torch.randn(6, group.generator_shape[0], generator=torch.Generator().manual_seed(1), dtype=F64)
    U = torch.linalg.matrix_exp(torch.einsum("na,aij->nij", phi, group.generators))
    assert float(U[:, :4, 4:].abs().max()) == 0.0
    for h in range(2):
        block = U[:, 4 * h:4 * h + 4, 4 * h:4 * h + 4]
        det = torch.linalg.det(block)
        if name in DET_ONE:
            torch.testing.assert_close(det, torch.ones_like(det))
        if name != "block_sl":
            assert float(torch.tril(block, diagonal=-1).abs().max()) < 1e-12
            diag = block.diagonal(dim1=-2, dim2=-1)
            if name == "block_unipotent":
                torch.testing.assert_close(diag, torch.ones_like(diag))
            else:
                assert bool((diag > 0).all()) and float((det - 1.0).abs().max()) > 1e-3   # a genuine volume
    if name == "block_unipotent":                                                     # e^X is a polynomial
        X = torch.einsum("na,aij->nij", phi, group.generators)[:, :4, :4]
        series = sum(torch.linalg.matrix_power(X, k) / math.factorial(k) for k in range(4))
        torch.testing.assert_close(U[:, :4, :4], series, atol=1e-12, rtol=0)


@pytest.mark.parametrize("name", DET_ONE)
def test_trace_projection_is_the_identity(name: str) -> None:
    group = get_group(name)(6, 2, dtype=F64)
    assert float(group.block_trace_vectors().abs().max()) < 1e-15
    phi = torch.randn(5, group.generator_shape[0], generator=torch.Generator().manual_seed(2), dtype=F64)
    torch.testing.assert_close(project_phi_to_slk(phi, group, group.irrep_dims), phi, atol=0, rtol=0)


@pytest.mark.parametrize("name", GROUPS)
def test_admissibility_with_the_diagonal_family_as_negative_control(name: str) -> None:
    group = get_group(name)(6, 2, dtype=F64)
    assert check_admissible(group, "gaussian_full")
    assert not check_admissible(group, "gaussian_diagonal")
    assert group.invariant_for("gaussian_full") and not group.invariant_for("gaussian_diagonal")


def test_config_accepts_refuses_and_warns_on_rope() -> None:
    for name in GROUPS:
        cfg = test_config(gauge_group=name, embed_dim=6, n_heads=2, rope_mode="none")
        assert cfg.gauge_tying == "per_head" and cfg.head_dims == [3, 3]
        with pytest.raises(ValueError, match="gauge_tying"):
            test_config(gauge_group=name, embed_dim=6, n_heads=2, gauge_tying="shared")
        with pytest.raises(ValueError, match="divisible"):
            test_config(gauge_group=name, embed_dim=7, n_heads=2)
        with pytest.raises(ValueError, match="irrep-labeled tower"):
            test_config(gauge_group=name, embed_dim=6, n_heads=2, use_cg_coupling=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            test_config(gauge_group=name, embed_dim=6, n_heads=2, rope_mode="frame_value")
        left = [w for w in caught if "leaves the structure group" in str(w.message)]
        assert bool(left) == (name != "block_sl"), name
    for name in ("block_sl", "block_unipotent"):
        with pytest.raises(ValueError, match="head blocks of size >= 2"):
            test_config(gauge_group=name, embed_dim=4, n_heads=4)
    assert test_config(gauge_group="block_borel", embed_dim=4, n_heads=4, rope_mode="none").head_dims == [1] * 4
    assert test_config(gauge_group="block_sl", embed_dim=6, n_heads=2, phi_frame_condition_max=20.0)
    for name in ("block_borel", "block_unipotent"):
        with pytest.raises(ValueError, match="phi_frame_condition_max"):
            test_config(gauge_group=name, embed_dim=6, n_heads=2, rope_mode="none", phi_frame_condition_max=20.0)


def test_frame_condition_bound_on_block_sl() -> None:
    from vfe3.gauge_optim import bound_phi_frame_condition_
    model = VFEModel(_quiet_config(gauge_group="block_sl", embed_dim=6, n_heads=2, vocab_size=6, max_seq_len=3,
                                   pos_phi="none"))
    with torch.no_grad():
        model.prior_bank.phi_embed.normal_(mean=0.0, std=2.0, generator=torch.Generator().manual_seed(5))
    bound_phi_frame_condition_(model, 20.0)
    U = torch.linalg.matrix_exp(model.group.embed_algebra(model.prior_bank.phi_embed.detach().double(),
                                                          dtype=F64))
    for h in range(2):
        block = U[:, 3 * h:3 * h + 3, 3 * h:3 * h + 3]
        s = torch.linalg.svdvals(block)
        assert float((s[:, 0] / s[:, -1]).max()) <= 20.0 * (1 + 1e-5)
        det = torch.linalg.det(block)
        torch.testing.assert_close(det, torch.ones_like(det), atol=1e-5, rtol=0)


@pytest.mark.parametrize("name", GROUPS)
@pytest.mark.parametrize("route", sorted(ROUTES))
def test_train_step_and_gauge_equivariance(name: str, route: str) -> None:
    from vfe3.train import build_optimizer, train_step
    from vfe3.viz import extract
    model = _model(name, **ROUTES[route])
    assert model.prior_bank.phi_embed.shape[-1] == 2 * N_GEN[name](3)
    report = extract.projected_gauge_equivariance(model.eval(), torch.tensor([[1, 4, 2]]), n_draws=2)
    residuals = report["residuals"]
    assert residuals and all(max(v) < 1e-10 for v in residuals.values()), residuals
    if "receiver_energy" in residuals:
        if name in DET_ONE:
            assert report["log_det_shift_max"] < 1e-12                         # the volume does not move
        else:
            assert report["log_det_shift_max"] > 1e-6
    model.train()
    optimizer = build_optimizer(model, model.cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    phi0 = model.prior_bank.phi_embed.detach().clone()
    loss = train_step(model, optimizer, scheduler, torch.tensor([[1, 4, 2]]), torch.tensor([[4, 2, 3]]))
    assert math.isfinite(loss)
    assert float((model.prior_bank.phi_embed.detach() - phi0).abs().max()) > 0.0


@pytest.mark.parametrize("name", GROUPS)
def test_pullback_group_train_step(name: str) -> None:
    from vfe3.train import build_optimizer, train_step
    model = _model(name, m_phi_update_mode="pullback_group", m_phi_group_retraction="chart",
                   transport_chart_max_norm=6.0, m_phi_lr=0.1, pos_phi="none")
    optimizer = build_optimizer(model, model.cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    phi0 = model.prior_bank.phi_embed.detach().clone()
    loss = train_step(model, optimizer, scheduler, torch.tensor([[1, 4, 2]]), torch.tensor([[4, 2, 3]]))
    assert math.isfinite(loss)
    assert float((model.prior_bank.phi_embed.detach() - phi0).abs().max()) > 0.0


def test_block_sl_rope_train_step() -> None:
    from vfe3.train import build_optimizer, train_step
    model = _model("block_sl", rope_mode="frame", n_layers=2)
    optimizer = build_optimizer(model, model.cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    assert math.isfinite(train_step(model, optimizer, scheduler, torch.tensor([[1, 4, 2]]),
                                    torch.tensor([[4, 2, 3]])))


@pytest.mark.parametrize("name", GROUPS)
@pytest.mark.parametrize("route", ["predictive", "gradient"])
def test_phi_gradient_matches_finite_difference(name: str, route: str) -> None:
    model = _model(name, F64, **ROUTES[route]).eval()
    tokens = torch.tensor([[1, 4, 2]])
    phi = model.prior_bank.phi_embed
    w_mu = torch.randn(1, 3, 6, generator=torch.Generator().manual_seed(3), dtype=F64)

    def objective() -> torch.Tensor:
        belief, _ = model.forward_beliefs(tokens)
        w_sigma = torch.randn(belief.sigma.shape, generator=torch.Generator().manual_seed(4), dtype=F64)
        return (w_mu * belief.mu).sum() + (w_sigma * belief.sigma).sum()

    objective().backward()
    grad = phi.grad.clone()
    h = 1e-5
    last = phi.shape[-1] - 1                                  # block_sl: the second head's last Helmert diagonal
    for row, col in [(1, 0), (4, 1), (2, last)]:
        with torch.no_grad():
            phi[row, col] += h
            plus = float(objective())
            phi[row, col] -= 2 * h
            minus = float(objective())
            phi[row, col] += h
        fd = (plus - minus) / (2 * h)
        assert abs(fd - float(grad[row, col])) <= 1e-7 * max(1.0, abs(fd)), (row, col, fd, grad[row, col])
