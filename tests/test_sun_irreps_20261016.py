r"""``su_n`` M1 (gauge-group-buildouts-2026-10-05.md, section 2): SU(N) irrep towers, trivial and adjoint blocks.

The basis is :math:`X_a = i\lambda_a` (generalized Gell-Mann), stored as its real 2N x 2N realification; the adjoint
'pq1_1' is :math:`\mathrm{ad}(X_a)_{cb} = f_{abc}`. Pins, on CPU at tiny size in float64: the basis (anti-Hermitian,
traceless, Gram 2I, so(N) as the leading slice, real structure constants); the closed-form real dimensions against
the Weyl formula; the adjoint block (exact skew, homomorphism, Casimir 4 C_2 = 4N, commutant dimension 1 = real
type); the builder (Gram certificate, admissibility, tying, deterministic fingerprint); config acceptance and
refusals; ``so_n`` bit-identical after the tower refactor (fingerprints recorded on unmodified main); and projected
train steps, the gauge-equivariance residual and a finite-difference phi gradient.
"""
import math
import warnings

import pytest
import torch

from tests.profiles import test_config, tiny_config
from vfe3.geometry.generators import generate_son, generate_sun, generate_sun_real, realify
from vfe3.geometry.groups import check_admissible, get_group, group_traits
from vfe3.geometry.irreps import irrep_dim, irrep_generators, structure_constants
from vfe3.model.model import VFEModel

F64 = torch.float64
ROUTES = {
    "predictive": {},
    "predictive_full": dict(family="gaussian_full", decode_mode="full_chunked"),
    "gradient": dict(attention_mode="projected_gradient", predictive_covariance="passthrough",
                     attention_residual=False),
}


@pytest.mark.parametrize("N", [3, 4])
def test_basis_properties(N: int) -> None:
    X = generate_sun(N)                                                          # (N^2 - 1, N, N) complex
    assert X.shape == (N * N - 1, N, N)
    torch.testing.assert_close(X + X.conj().transpose(-1, -2), torch.zeros_like(X))
    torch.testing.assert_close(X.diagonal(dim1=-2, dim2=-1).sum(-1), torch.zeros(N * N - 1, dtype=X.dtype))
    gram = torch.einsum("aij,bij->ab", X.conj(), X)
    torch.testing.assert_close(gram, 2.0 * torch.eye(N * N - 1, dtype=X.dtype))
    m = N * (N - 1) // 2
    torch.testing.assert_close(X[:m].real, generate_son(N, dtype=F64))
    f = structure_constants(generate_sun_real(N))                                # real by construction
    for a in range(N * N - 1):
        for b in range(N * N - 1):
            lhs = X[a] @ X[b] - X[b] @ X[a]
            torch.testing.assert_close(lhs, torch.einsum("c,cij->ij", f[a, b].to(X.dtype), X), atol=1e-12, rtol=0)
    R = generate_sun_real(N)
    assert float((R + R.transpose(-1, -2)).abs().max()) == 0.0
    torch.testing.assert_close(realify(X[0] @ X[1]), realify(X[0]) @ realify(X[1]))


def _weyl_su3(p: int, q: int) -> int:
    return (p + 1) * (q + 1) * (p + q + 2) // 2


def test_real_dimensions_against_the_weyl_formula() -> None:
    for p, q in [(1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (3, 0), (3, 3)]:
        D = _weyl_su3(p, q)
        assert irrep_dim(3, algebra="su", label=f"pq{p}_{q}") == (D if p == q else 2 * D)
    assert [irrep_dim(3, algebra="su", label=l) for l in ("pq1_0", "pq1_1", "pq2_0", "pq2_2")] == [6, 8, 12, 27]
    assert irrep_dim(4, algebra="su", label="pq1_1") == 15 and irrep_dim(5, algebra="su", label="pq1_1") == 24
    assert irrep_dim(3, algebra="su", label="pq0_0") == 1
    assert irrep_dim(3, algebra="so", label="l2") == 5                           # so labels unchanged


@pytest.mark.parametrize("N", [3, 4])
def test_adjoint_block(N: int) -> None:
    G_def = generate_sun_real(N)
    rho = irrep_generators(G_def, algebra="su", label="pq1_1")                   # (n, n, n)
    n = N * N - 1
    assert rho.shape == (n, n, n)
    assert float((rho + rho.transpose(-1, -2)).abs().max()) == 0.0              # exactly skew
    f = structure_constants(G_def)
    for a in range(n):
        for b in range(n):
            torch.testing.assert_close(rho[a] @ rho[b] - rho[b] @ rho[a],
                                       torch.einsum("c,cij->ij", f[a, b], rho), atol=1e-12, rtol=0)
    casimir = -(rho @ rho).sum(0)
    torch.testing.assert_close(casimir, 4.0 * N * torch.eye(n, dtype=F64), atol=1e-12, rtol=0)
    # Real commutant: matrices C with C rho_a = rho_a C for every a; dimension 1 means real type.
    eye = torch.eye(n, dtype=F64)
    system = torch.cat([torch.kron(eye, r.contiguous()) - torch.kron(r.T.contiguous(), eye) for r in rho])
    sv = torch.linalg.svdvals(system)
    assert int((sv < 1e-9 * sv.max()).sum()) == 1


def test_builder_certificate_admissibility_and_tying() -> None:
    group = get_group("su_n")(17, group_n=3, irrep_spec=[("pq1_1", 2), ("pq0_0", 1)], dtype=F64)
    assert group.generator_shape == (8, 17, 17) and group.irrep_dims == [8, 8, 1]
    assert group.skew_symmetric and group.algebra == "su" and group.tying == "shared"
    assert group.irrep_labels == ["pq1_1", "pq1_1", "pq0_0"]
    assert group.frobenius_gram_diagonal and group.frobenius_gram_uniform == pytest.approx(24.0)
    assert group.pullback_capability is None
    assert check_admissible(group, "gaussian_full") and not check_admissible(group, "gaussian_diagonal")
    again = get_group("su_n")(17, group_n=3, irrep_spec=[("pq1_1", 2), ("pq0_0", 1)], dtype=F64)
    assert group.basis_fingerprint == again.basis_fingerprint                    # deterministic rebuild
    # Pinned: the learned tables of an su_n checkpoint are tied to this basis (rebuilt at every load).
    assert group.basis_fingerprint == "aab0b8e25e9000707f0ff03392bf71a5aa17e3bf34b68f9d4fe5c7bd46d0e6ee"
    untied = get_group("su_n")(17, group_n=3, irrep_spec=[("pq1_1", 2), ("pq0_0", 1)], tying="per_head")
    assert untied.generator_shape == (16, 17, 17) and untied.tying == "per_head"
    traits = group_traits("su_n")
    assert traits.algebra == "su" and traits.native_tying == "shared" and not traits.contains_pair_rotations


def test_config_accepts_and_refuses() -> None:
    cfg = test_config(gauge_group="su_n", group_n=3, irrep_spec=[("pq1_1", 2)], embed_dim=16, n_heads=2)
    assert cfg.gauge_tying == "shared" and cfg.head_dims == [8, 8]
    base = dict(gauge_group="su_n", group_n=3, embed_dim=16, n_heads=2)
    with pytest.raises(ValueError, match="group_n >= 3"):
        test_config(**{**base, "group_n": 2, "irrep_spec": [("pq1_1", 2)], "embed_dim": 6})
    with pytest.raises(ValueError, match="p >= q"):
        test_config(**{**base, "irrep_spec": [("pq1_2", 2)], "embed_dim": 60})
    with pytest.raises(ValueError, match="no construction yet"):
        test_config(**{**base, "irrep_spec": [("pq1_0", 2)], "embed_dim": 12})
    with pytest.raises(ValueError, match="so\\(N\\) towers only"):
        test_config(**{**base, "irrep_spec": "random"})
    with pytest.raises(ValueError, match="not built for su_n"):
        test_config(**{**base, "irrep_spec": [("pq1_1", 2)], "use_cg_coupling": True})
    with pytest.raises(ValueError, match="unknown irrep label"):
        test_config(**{**base, "irrep_spec": [("l1", 2)], "embed_dim": 6})


# Fingerprints of the float64 so_n generators recorded on unmodified main (f0df4b9 + #117) before the tower refactor.
SO_N_FINGERPRINTS = [
    ([("l1", 2), ("l2", 1)], 3, 11, "shared",
     "f6dfdbf6cf69d5fa88395bb0c9274460a518e0ef499082266abd84e0ae542464", (3, 11, 11)),
    ([("l1", 2), ("l2", 1)], 3, 11, "per_head",
     "d223cbe2a9aeb8ec9214dfc47dd6648ea7ed92aaf6518c3e552e4e5a5dbae183", (9, 11, 11)),
    ([("l1", 1), ("w2", 1)], 5, 15, "shared",
     "9027cd977fb58d7410b3d805babb9e11d6708da90e00d7d23a623bf112b1c480", (10, 15, 15)),
]


@pytest.mark.parametrize("spec, N, K, tying, fingerprint, shape", SO_N_FINGERPRINTS)
def test_so_n_is_bit_identical_after_the_tower_refactor(spec, N, K, tying, fingerprint, shape) -> None:
    group = get_group("so_n")(K, group_n=N, irrep_spec=spec, tying=tying)
    assert group.basis_fingerprint == fingerprint and group.generator_shape == shape and group.name == "so_n"


def _model(dtype: torch.dtype = torch.float32, **overrides: object) -> VFEModel:
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(tiny_config(gauge_group="su_n", group_n=3, irrep_spec=[("pq1_1", 2)],
                                     embed_dim=16, n_heads=2, **overrides)).to(dtype)
    with torch.no_grad():
        model.prior_bank.phi_embed.normal_(mean=0.0, std=0.3)
    return model


@pytest.mark.parametrize("route", sorted(ROUTES))
@pytest.mark.parametrize("tying", ["shared", "per_head"])
def test_train_step_and_gauge_equivariance(route: str, tying: str) -> None:
    from vfe3.train import build_optimizer, train_step
    from vfe3.viz import extract
    model = _model(gauge_tying=tying, **ROUTES[route])
    assert model.prior_bank.phi_embed.shape[-1] == (8 if tying == "shared" else 16)
    residuals = extract.projected_gauge_equivariance(
        model.eval(), torch.tensor([[1, 4, 2]]), n_draws=2)["residuals"]
    assert residuals and all(max(v) < 1e-9 for v in residuals.values()), residuals
    model.train()
    optimizer = build_optimizer(model, model.cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    phi0 = model.prior_bank.phi_embed.detach().clone()
    loss = train_step(model, optimizer, scheduler, torch.tensor([[1, 4, 2]]), torch.tensor([[4, 2, 3]]))
    assert math.isfinite(loss)
    assert float((model.prior_bank.phi_embed.detach() - phi0).abs().max()) > 0.0


@pytest.mark.parametrize("route", ["predictive", "gradient"])
def test_phi_gradient_matches_finite_difference(route: str) -> None:
    model = _model(F64, **ROUTES[route]).eval()
    tokens = torch.tensor([[1, 4, 2]])
    phi = model.prior_bank.phi_embed
    w_mu = torch.randn(1, 3, 16, generator=torch.Generator().manual_seed(3), dtype=F64)

    def objective() -> torch.Tensor:
        belief, _ = model.forward_beliefs(tokens)
        w_sigma = torch.randn(belief.sigma.shape, generator=torch.Generator().manual_seed(4), dtype=F64)
        return (w_mu * belief.mu).sum() + (w_sigma * belief.sigma).sum()

    objective().backward()
    grad = phi.grad.clone()
    h = 1e-5
    for row, col in [(1, 0), (4, 4), (2, 7)]:                                   # so(3) slice, symmetric, diagonal
        with torch.no_grad():
            phi[row, col] += h
            plus = float(objective())
            phi[row, col] -= 2 * h
            minus = float(objective())
            phi[row, col] += h
        fd = (plus - minus) / (2 * h)
        assert abs(fd - float(grad[row, col])) <= 1e-7 * max(1.0, abs(fd)), (row, col, fd, grad[row, col])
