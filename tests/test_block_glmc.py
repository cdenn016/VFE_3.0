r"""``block_glmc``: GL(m, C)^H realified on d_head = 2m (gauge-group-buildouts-2026-10-05.md, section 1, M1).

Complex coordinate k is the real pair (2k, 2k + 1) with J = I_m (x) [[0, -1], [1, 0]]. Pins, on CPU at tiny size
in float64: the generator entries and order, [G_a, J] = 0, Gram = 2I and the diagonal-Gram norm route, bracket
closure; exp equal to the realified complex exponential and commuting with J; admissibility for the full
Gaussian; config acceptance and refusals; RoPE blocks commuting with J; the trace projection fixing |det_C| = 1 and
keeping the phase; the circularity and complex-phase diagnostics; and train steps with and without RoPE, the
gauge-equivariance residual and a finite-difference phi gradient.
"""
import math
import warnings

import pytest
import torch

from tests.profiles import test_config, tiny_config
from vfe3.geometry.groups import check_admissible, get_group, group_traits
from vfe3.geometry.lie_ops import project_phi_to_slk
from vfe3.geometry.rope import build_rope_rotation
from vfe3.metrics import _complex_structure, per_head_circularity, per_head_complex_phase
from vfe3.model.model import VFEModel

F64 = torch.float64
ROUTES = {
    "predictive": {},
    "predictive_full": dict(family="gaussian_full", decode_mode="full_chunked"),
    "gradient": dict(attention_mode="projected_gradient", predictive_covariance="passthrough",
                     attention_residual=False),
}


def _J(K: int) -> torch.Tensor:
    return _complex_structure(K, dtype=F64, device=torch.device("cpu"))


def _realify(Z: torch.Tensor) -> torch.Tensor:
    r"""(m, m) complex -> (2m, 2m) real in the interleaved layout: A (x) I_2 + B (x) J."""
    J2 = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=F64)
    return torch.kron(Z.real, torch.eye(2, dtype=F64)) + torch.kron(Z.imag, J2)


def _model(dtype: torch.dtype = torch.float32, **overrides: object) -> VFEModel:
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(tiny_config(gauge_group="block_glmc", embed_dim=8, n_heads=2, **overrides)).to(dtype)
    with torch.no_grad():
        model.prior_bank.phi_embed.normal_(mean=0.0, std=0.3)
    return model


def test_generator_entries_order_and_complex_structure() -> None:
    group = get_group("block_glmc")(8, 2, dtype=F64)
    G = group.generators
    assert G.shape == (2 * 2 * 2 ** 2, 8, 8) and group.irrep_dims == [4, 4] and group.tying == "per_head"
    # 2026-10-07 (audit F47): every registered group carries the pullback_per_block frame M-step.
    assert not group.skew_symmetric and group.pullback_capability == "pullback_per_block" and group.algebra is None
    assert set(G.unique().tolist()) <= {-1.0, 0.0, 1.0}
    J = _J(8)
    assert float((G @ J - J @ G).abs().max()) == 0.0                     # [G_a, J] = 0
    # Coordinates are (Re, Im) of the complex algebra element, order (h, re/im, i, j).
    Z = torch.complex(torch.randn(2, 2, dtype=F64), torch.randn(2, 2, dtype=F64))
    coords = torch.cat([Z.real.reshape(-1), Z.imag.reshape(-1)])
    torch.testing.assert_close(torch.einsum("a,aij->ij", coords, G[:8])[:4, :4], _realify(Z))
    traits = group_traits("block_glmc")
    assert traits.min_head_dim == 2 and traits.head_dim_multiple == 2 and traits.contains_pair_rotations


def test_gram_is_twice_identity_and_bracket_closure() -> None:
    group = get_group("block_glmc")(8, 2, dtype=F64)
    G = group.generators
    n = G.shape[0]
    torch.testing.assert_close(torch.einsum("aij,bij->ab", G, G), 2.0 * torch.eye(n, dtype=F64))
    assert group.gram_diagonal_uniform() == 2.0 and group.phi_norm_route() == "diagonal_gram"
    flat = G.reshape(n, -1) / 2.0 ** 0.5
    for a in range(n):
        for b in range(n):
            c = (G[a] @ G[b] - G[b] @ G[a]).reshape(-1)
            assert float((c - flat.T @ (flat @ c)).abs().max()) < 1e-12


def test_exp_is_realified_complex_exp_and_commutes_with_J() -> None:
    group = get_group("block_glmc")(8, 2, dtype=F64)
    phi = torch.randn(group.generator_shape[0], generator=torch.Generator().manual_seed(1), dtype=F64)
    U = torch.linalg.matrix_exp(group.embed_algebra(phi))
    J = _J(8)
    torch.testing.assert_close(U @ J, J @ U)
    for h in range(2):
        p = phi[8 * h:8 * h + 8]
        Z = torch.complex(p[:4].reshape(2, 2), p[4:].reshape(2, 2))
        torch.testing.assert_close(U[4 * h:4 * h + 4, 4 * h:4 * h + 4], _realify(torch.linalg.matrix_exp(Z)))
    assert float(U[:4, 4:].abs().max()) == 0.0


def test_admissibility_with_the_diagonal_family_as_negative_control() -> None:
    group = get_group("block_glmc")(4, 2, dtype=F64)
    assert check_admissible(group, "gaussian_full")
    assert not check_admissible(group, "gaussian_diagonal")


def test_config_accepts_and_refuses() -> None:
    cfg = test_config(gauge_group="block_glmc", embed_dim=8, n_heads=2)
    assert cfg.gauge_tying == "per_head" and cfg.head_dims == [4, 4]
    with pytest.raises(ValueError, match="multiple of 2"):
        test_config(gauge_group="block_glmc", embed_dim=6, n_heads=2)
    with pytest.raises(ValueError, match="head blocks of size >= 2"):
        test_config(gauge_group="block_glmc", embed_dim=4, n_heads=4)
    with pytest.raises(ValueError, match="gauge_tying"):
        test_config(gauge_group="block_glmc", embed_dim=8, n_heads=2, gauge_tying="shared")
    with pytest.raises(ValueError, match="irrep-tower"):
        test_config(gauge_group="block_glmc", embed_dim=8, n_heads=2, group_n=3)
    with pytest.raises(ValueError, match="irrep-labeled tower"):
        test_config(gauge_group="block_glmc", embed_dim=8, n_heads=2, use_cg_coupling=True)
    # The pullback frame M-step reaches block_glmc since 2026-10-07 (audit F47).
    test_config(gauge_group="block_glmc", embed_dim=8, n_heads=2, m_phi_update_mode="pullback_group")
    with pytest.raises(ValueError, match="block_glk"):
        test_config(gauge_group="block_glmc", embed_dim=8, n_heads=2, m_phi_update_mode="fisher_group")
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                   # RoPE stays inside the group
        test_config(gauge_group="block_glmc", embed_dim=8, n_heads=2, pos_rotation="rope")
    with pytest.raises(ValueError, match="even head size"):
        get_group("block_glmc")(6, 2)


def test_rope_blocks_and_folded_frame_commute_with_J() -> None:
    R = build_rope_rotation(torch.arange(5), [4, 4], dtype=F64)               # (5, 8, 8)
    J = _J(8)
    torch.testing.assert_close(R @ J, J @ R)
    group = get_group("block_glmc")(8, 2, dtype=F64)
    phi = torch.randn(5, group.generator_shape[0], generator=torch.Generator().manual_seed(2), dtype=F64)
    V = torch.linalg.matrix_exp(group.embed_algebra(phi)) @ R.transpose(-1, -2)
    torch.testing.assert_close(V @ J, J @ V)


def test_trace_projection_fixes_complex_modulus_and_keeps_phase() -> None:
    group = get_group("block_glmc")(8, 2, dtype=F64)
    phi = torch.randn(3, group.generator_shape[0], generator=torch.Generator().manual_seed(3), dtype=F64)
    projected = project_phi_to_slk(phi, group.generators, group.irrep_dims)
    U = torch.linalg.matrix_exp(group.embed_algebra(phi))
    U0 = torch.linalg.matrix_exp(group.embed_algebra(projected))
    for h in range(2):
        block = U0[:, 4 * h:4 * h + 4, 4 * h:4 * h + 4]
        torch.testing.assert_close(torch.linalg.det(block), torch.ones(3, dtype=F64))
    torch.testing.assert_close(per_head_complex_phase(U0, [4, 4]), per_head_complex_phase(U, [4, 4]))
    # Phase is Im tr of the head's complex algebra element (mod 2 pi): sum of its diagonal Im coordinates.
    p = phi[:, 4:8]
    expected = torch.remainder(p[:, 0] + p[:, 3] + math.pi, 2 * math.pi) - math.pi
    torch.testing.assert_close(per_head_complex_phase(U, [4, 4])[:, 0], expected)


def test_circularity_range_invariance_and_closed_form() -> None:
    gen = torch.Generator().manual_seed(4)
    A = torch.randn(5, 8, 8, generator=gen, dtype=F64)
    sigma = A @ A.transpose(-1, -2) + 0.5 * torch.eye(8, dtype=F64)
    c = per_head_circularity(sigma, [4, 4])
    assert c.shape == (5, 2, 4) and bool((c >= 0).all()) and bool((c < 1).all()) and float(c.max()) > 1e-2
    group = get_group("block_glmc")(8, 2, dtype=F64)
    phi = torch.randn(group.generator_shape[0], generator=gen, dtype=F64)
    g = torch.linalg.matrix_exp(group.embed_algebra(phi))
    torch.testing.assert_close(per_head_circularity(g @ sigma @ g.T, [4, 4]), c)
    J = _J(8)
    proper = 0.5 * (sigma + J.T @ sigma @ J)
    assert float(per_head_circularity(proper, [4, 4]).max()) < 1e-12
    s = torch.tensor([1.0, 3.0, 2.0, 2.0, 5.0, 1.0, 0.5, 4.0], dtype=F64)
    closed = (s[0::2] - s[1::2]).abs() / (s[0::2] + s[1::2])
    got = per_head_circularity(torch.diag(s), [4, 4])
    for h in range(2):
        torch.testing.assert_close(got[h], closed[2 * h:2 * h + 2].repeat_interleave(2).sort(descending=True).values)


@pytest.mark.parametrize("route", sorted(ROUTES))
def test_train_step_and_gauge_equivariance(route: str) -> None:
    from vfe3.train import build_optimizer, train_step
    from vfe3.viz import extract
    model = _model(**ROUTES[route])
    assert model.prior_bank.phi_embed.shape[-1] == 2 * 2 * 2 ** 2
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


def test_rope_train_step() -> None:
    from vfe3.train import build_optimizer, train_step
    model = _model(pos_rotation="rope", rope_on_value=False, n_layers=2)
    optimizer = build_optimizer(model, model.cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    assert math.isfinite(train_step(model, optimizer, scheduler, torch.tensor([[1, 4, 2]]),
                                    torch.tensor([[4, 2, 3]])))


@pytest.mark.parametrize("route", ["predictive", "gradient"])
def test_phi_gradient_matches_finite_difference(route: str) -> None:
    model = _model(F64, **ROUTES[route]).eval()
    tokens = torch.tensor([[1, 4, 2]])
    phi = model.prior_bank.phi_embed
    w_mu = torch.randn(1, 3, 8, generator=torch.Generator().manual_seed(3), dtype=F64)

    def objective() -> torch.Tensor:
        belief, _ = model.forward_beliefs(tokens)
        w_sigma = torch.randn(belief.sigma.shape, generator=torch.Generator().manual_seed(4), dtype=F64)
        return (w_mu * belief.mu).sum() + (w_sigma * belief.sigma).sum()

    objective().backward()
    grad = phi.grad.clone()
    h = 1e-5
    for row, col in [(1, 0), (4, 5), (2, phi.shape[-1] - 1)]:                # Re, Im, second head's Im
        with torch.no_grad():
            phi[row, col] += h
            plus = float(objective())
            phi[row, col] -= 2 * h
            minus = float(objective())
            phi[row, col] += h
        fd = (plus - minus) / (2 * h)
        assert abs(fd - float(grad[row, col])) <= 1e-7 * max(1.0, abs(fd)), (row, col, fd, grad[row, col])
