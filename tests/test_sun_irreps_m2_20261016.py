r"""``su_n`` M2 (gauge-group-buildouts-2026-10-05.md, section 2): every SU(N) irrep (p, q), and CG coupling.

Complex-type irreps (p != q) are the realification of the action on the traceless tensors of Sym^p (x) Sym^q, of real
dimension 2D; real-type irreps (p = q >= 2) get a real form of dimension D from the slot-swap involution. Pins, on
CPU at tiny size in float64: dimensions, the homomorphism, the Casimir 4 C_2, the real commutant dimension (2 for
complex type, 1 for real type), the centre Z_N acting as omega^(p-q), a deterministic tower fingerprint, CG
multiplicities (8 (x) 8 -> 8 twice, f and d) with the self-pair keeping only the symmetric d slot, config
acceptance, and train steps, the gauge-equivariance residual and a finite-difference phi gradient on a mixed tower.
"""
import math
import subprocess
import sys
import warnings

import pytest
import torch

from tests.profiles import test_config, tiny_config
from vfe3.geometry.cg import cg_intertwiners
from vfe3.geometry.generators import generate_sun, generate_sun_real
from vfe3.geometry.groups import get_group
from vfe3.geometry.irreps import (_assert_homomorphism, irrep_dim, irrep_generators, structure_constants,
                                  su_casimir)
from vfe3.model.cg_coupling import CGCoupling
from vfe3.model.model import VFEModel

F64 = torch.float64
SU3 = [("pq1_0", 6, 2), ("pq2_0", 12, 2), ("pq2_1", 30, 2), ("pq2_2", 27, 1), ("pq3_0", 20, 2)]


def _commutant_dim(rho: torch.Tensor) -> int:
    n = rho.shape[-1]
    eye = torch.eye(n, dtype=F64)
    system = torch.cat([torch.kron(eye, r.contiguous()) - torch.kron(r.T.contiguous(), eye) for r in rho])
    sv = torch.linalg.svdvals(system)
    return int((sv < 1e-9 * sv.max()).sum())


@pytest.mark.parametrize("label, dim, commutant", SU3)
def test_su3_irreps(label: str, dim: int, commutant: int) -> None:
    G = generate_sun_real(3)
    rho = irrep_generators(G, algebra="su", label=label)
    assert rho.shape == (8, dim, dim) and irrep_dim(3, algebra="su", label=label) == dim
    assert float((rho + rho.transpose(-1, -2)).abs().max()) == 0.0
    _assert_homomorphism(rho, structure_constants(G), atol=1e-10, label=label)
    p, q = int(label[2]), int(label[4])
    torch.testing.assert_close(-(rho @ rho).sum(0), 4.0 * su_casimir(3, p, q) * torch.eye(dim, dtype=F64),
                               atol=1e-9, rtol=0)
    assert _commutant_dim(rho) == commutant                                     # 2 complex type, 1 real type


def test_casimir_values() -> None:
    assert su_casimir(3, 1, 0) == pytest.approx(4 / 3) and su_casimir(3, 1, 1) == pytest.approx(3.0)
    assert su_casimir(3, 2, 0) == pytest.approx(10 / 3) and su_casimir(3, 2, 2) == pytest.approx(8.0)
    assert su_casimir(4, 1, 0) == pytest.approx(15 / 8) and su_casimir(5, 1, 1) == pytest.approx(5.0)


@pytest.mark.parametrize("label", ["pq1_0", "pq2_0", "pq2_1", "pq1_1", "pq2_2"])
def test_centre_acts_as_omega_to_the_p_minus_q(label: str) -> None:
    N = 3
    X = generate_sun(N)                                                          # (8, 3, 3), tr(X^H X) = 2 delta
    diag = torch.full((N,), 1.0, dtype=F64)
    diag[-1] = -(N - 1)
    centre = torch.diag((2j * math.pi / N) * diag.to(torch.complex128))         # exp(centre) = omega I
    torch.testing.assert_close(torch.linalg.matrix_exp(centre),
                               torch.exp(torch.tensor(2j * math.pi / N)) * torch.eye(N, dtype=torch.complex128))
    coords = torch.einsum("aij,ij->a", X.conj(), centre).real / 2.0
    rho = irrep_generators(generate_sun_real(N), algebra="su", label=label)
    g = torch.linalg.matrix_exp(torch.einsum("a,aij->ij", coords, rho))
    p, q = int(label[2]), int(label[4])
    if p == q:
        expected = torch.eye(rho.shape[-1], dtype=F64)
    else:
        theta = 2 * math.pi * (p - q) / N
        D = rho.shape[-1] // 2
        c, s = math.cos(theta) * torch.eye(D, dtype=F64), math.sin(theta) * torch.eye(D, dtype=F64)
        expected = torch.cat([torch.cat([c, -s], dim=1), torch.cat([s, c], dim=1)], dim=0)
    torch.testing.assert_close(g, expected, atol=1e-9, rtol=0)


def test_su4_fundamental_and_size_guard() -> None:
    rho = irrep_generators(generate_sun_real(4), algebra="su", label="pq1_0")
    assert rho.shape == (15, 8, 8) and _commutant_dim(rho) == 2
    with pytest.raises(ValueError, match="exceeds the supported"):
        test_config(gauge_group="su_n", group_n=3, irrep_spec=[("pq4_3", 2)],
                    embed_dim=2 * irrep_dim(3, algebra="su", label="pq4_3"), n_heads=2)


def test_tower_fingerprint_is_deterministic_across_processes_and_threads() -> None:
    # The real-type (2, 2) images differed in the last bit at four threads before the build was single-threaded.
    code = ("import sys, torch; sys.path.insert(0, '.'); torch.set_num_threads({}); "
            "from vfe3.geometry.groups import get_group; "
            "print(get_group('su_n')(53, group_n=3, irrep_spec=[('pq1_0', 1), ('pq2_2', 1), ('pq2_0', 1), "
            "('pq1_1', 1)]).basis_fingerprint, torch.get_num_threads())")
    outputs = [subprocess.run([sys.executable, "-c", code.format(t)], capture_output=True, text=True,
                              check=True).stdout.split() for t in (1, 4, 4)]
    assert [int(threads) for _, threads in outputs] == [1, 4, 4]                # the caller's count is restored
    prints = {fingerprint for fingerprint, _ in outputs}
    here = get_group("su_n")(53, group_n=3, irrep_spec=[("pq1_0", 1), ("pq2_2", 1), ("pq2_0", 1), ("pq1_1", 1)])
    assert prints == {here.basis_fingerprint}


def test_cg_multiplicities_and_the_symmetric_self_pair() -> None:
    C = cg_intertwiners(3, algebra="su", label_a="pq1_1", label_b="pq1_1", label_c="pq1_1")
    assert C.shape[0] == 2                                                       # f and d
    assert cg_intertwiners(3, algebra="su", label_a="pq1_1", label_b="pq1_1", label_c="pq0_0").shape[0] == 1
    T = C.reshape(2, 8, 8, 8)                                                   # (slot, c, a, b)
    sym, anti = 0.5 * (T + T.transpose(2, 3)), 0.5 * (T - T.transpose(2, 3))
    # The null space holds exactly one swap-symmetric (d) and one swap-antisymmetric (f) direction; the solver's
    # basis inside it is arbitrary (cg.py), so the two slots may be any rotation of {d, f}.
    assert int(torch.linalg.matrix_rank(sym.reshape(2, -1), atol=1e-9)) == 1
    assert int(torch.linalg.matrix_rank(anti.reshape(2, -1), atol=1e-9)) == 1
    cg = CGCoupling(3, "su", [8, 8], ["pq1_1", "pq1_1"])
    self_slots = {path[-1] for path in cg.paths if path[0] == path[2]}           # same source copy twice
    live = {m for m in range(2) if float(sym[m].abs().max()) > 1e-9}            # slots with a d component
    assert self_slots == live and len(self_slots) >= 1                          # f alone is pruned, d is kept


MIXED = dict(gauge_group="su_n", group_n=3, irrep_spec=[("pq1_0", 1), ("pq1_1", 1)], embed_dim=14, n_heads=2)
ROUTES = {
    "predictive": {},
    "gradient": dict(attention_mode="projected_gradient", predictive_covariance="passthrough",
                     attention_residual=False),
}


def test_config_accepts_mixed_towers_and_cg() -> None:
    cfg = test_config(**MIXED)
    assert cfg.head_dims == [6, 8] and cfg.gauge_tying == "shared"
    cg = test_config(**{**MIXED, "irrep_spec": [("pq1_1", 2)], "embed_dim": 16, "use_cg_coupling": True})
    assert cg.use_cg_coupling


def _model(dtype: torch.dtype = torch.float32, **overrides: object) -> VFEModel:
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(tiny_config(**{**MIXED, **overrides})).to(dtype)
    with torch.no_grad():
        model.prior_bank.phi_embed.normal_(mean=0.0, std=0.3)
    return model


@pytest.mark.parametrize("route", sorted(ROUTES))
@pytest.mark.parametrize("tying", ["shared", "per_head"])
def test_mixed_tower_train_step_and_gauge_equivariance(route: str, tying: str) -> None:
    from vfe3.train import build_optimizer, train_step
    from vfe3.viz import extract
    model = _model(gauge_tying=tying, **ROUTES[route])
    residuals = extract.projected_gauge_equivariance(
        model.eval(), torch.tensor([[1, 4, 2]]), n_draws=2)["residuals"]
    assert residuals and all(max(v) < 1e-9 for v in residuals.values()), residuals
    model.train()
    optimizer = build_optimizer(model, model.cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    assert math.isfinite(train_step(model, optimizer, scheduler, torch.tensor([[1, 4, 2]]),
                                    torch.tensor([[4, 2, 3]])))


def test_cg_coupled_adjoint_tower_trains() -> None:
    from vfe3.train import build_optimizer, train_step
    model = _model(irrep_spec=[("pq1_1", 2)], embed_dim=16, use_cg_coupling=True)
    assert model.cg_coupling is not None
    optimizer = build_optimizer(model, model.cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    assert math.isfinite(train_step(model, optimizer, scheduler, torch.tensor([[1, 4, 2]]),
                                    torch.tensor([[4, 2, 3]])))


def test_phi_gradient_matches_finite_difference_on_the_mixed_tower() -> None:
    model = _model(F64).eval()
    tokens = torch.tensor([[1, 4, 2]])
    phi = model.prior_bank.phi_embed
    w_mu = torch.randn(1, 3, 14, generator=torch.Generator().manual_seed(3), dtype=F64)

    def objective() -> torch.Tensor:
        belief, _ = model.forward_beliefs(tokens)
        w_sigma = torch.randn(belief.sigma.shape, generator=torch.Generator().manual_seed(4), dtype=F64)
        return (w_mu * belief.mu).sum() + (w_sigma * belief.sigma).sum()

    objective().backward()
    grad = phi.grad.clone()
    h = 1e-5
    for row, col in [(1, 0), (4, 4), (2, 7)]:
        with torch.no_grad():
            phi[row, col] += h
            plus = float(objective())
            phi[row, col] -= 2 * h
            minus = float(objective())
            phi[row, col] += h
        fd = (plus - minus) / (2 * h)
        assert abs(fd - float(grad[row, col])) <= 1e-7 * max(1.0, abs(fd)), (row, col, fd, grad[row, col])
