"""Remediation of the 2026-10-08 audit geometry items (docs/audits/audit-2026-10-08.md).

Tiny float64/float32 CPU checks: the overflow-safe Loewner divided differences, the SPD certificate
floor/ceiling counters, the off-strip Psi-inverse accuracy gate, the generator-free 'l0' block in the
per-block pullback, the soft off-strip rejection on the fisher_group chart route, the lazy generator
basis in manifold staging, and the comparison-RoPE map invariants.
"""
from tests.profiles import test_config

import math
import warnings

import pytest
import torch
from torch import nn

F64 = torch.float64


def _cfg(**values):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return test_config(**values)


# Loewner divided differences (general-A-core-math#3, geometer#3) ---------------------------------

def _offdiag_adjoint(a: float, b: float, kind: str, lower: float, upper, dtype) -> float:
    from vfe3.geometry.retraction import _loewner_adjoint

    lam = torch.tensor([[a, b]], dtype=dtype)
    grad = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]], dtype=dtype)
    out = _loewner_adjoint(lam, torch.eye(2, dtype=dtype)[None], kind, lower, upper, grad)
    return float(out[0, 0, 1])


@pytest.mark.parametrize("a,b", [(-45.0, 45.0), (45.0, -45.0), (49.5, -49.5), (40.0, -40.0)])
def test_exp_divided_difference_is_finite_in_float32_for_wide_gaps(a: float, b: float) -> None:
    exact = (math.exp(a) - math.exp(b)) / (a - b)
    value32 = _offdiag_adjoint(a, b, "exp_bounded", -50.0, 50.0, torch.float32)
    value64 = _offdiag_adjoint(a, b, "exp_bounded", -50.0, 50.0, F64)
    assert math.isfinite(value32)
    assert abs(value32 - exact) <= 1e-5 * abs(exact)
    assert abs(value64 - exact) <= 1e-14 * abs(exact)


@pytest.mark.parametrize("a,b", [(0.3, 0.3 + 1e-7), (1.0, 2.0), (-3.0, 4.0), (10.0, 10.5)])
def test_exp_divided_difference_matches_the_quotient_in_float64(a: float, b: float) -> None:
    exact = math.exp(b) * math.expm1(a - b) / (a - b)
    assert abs(_offdiag_adjoint(a, b, "exp_bounded", -50.0, 50.0, F64) - exact) <= 1e-14 * abs(exact)


@pytest.mark.parametrize("a,b", [(1e-6, 100.0), (2e-6, 100.0), (1e-6, 30.0), (1.0, 1.0 + 1e-6)])
def test_log_divided_difference_is_finite_and_accurate_in_float32(a: float, b: float) -> None:
    mpmath = pytest.importorskip("mpmath")
    mpmath.mp.dps = 50
    exact = float((mpmath.log(mpmath.mpf(a)) - mpmath.log(mpmath.mpf(b))) / (mpmath.mpf(a) - mpmath.mpf(b)))
    value32 = _offdiag_adjoint(a, b, "log_floor", 1e-8, None, torch.float32)
    value64 = _offdiag_adjoint(a, b, "log_floor", 1e-8, None, F64)
    assert math.isfinite(value32)
    assert abs(value32 - exact) <= 1e-5 * abs(exact)
    assert abs(value64 - exact) <= 1e-12 * abs(exact)


def test_exp_bounded_spectral_map_gradient_matches_finite_differences() -> None:
    from vfe3.geometry.retraction import _symmetric_spectral_map

    torch.manual_seed(0)
    A = torch.randn(3, 3, dtype=F64)
    A = (2.0 * (A + A.T)).requires_grad_(True)

    def f(M: torch.Tensor) -> torch.Tensor:
        return _symmetric_spectral_map(0.5 * (M + M.T), "exp_bounded", lower=-50.0, upper=50.0)

    assert torch.autograd.gradcheck(f, (A,), eps=1e-6, atol=1e-7, rtol=1e-6)


# SPD certificate counters (geometer#4) ------------------------------------------------------------

def test_ceiling_contacts_are_not_counted_as_floor_repairs() -> None:
    from vfe3.geometry.retraction import (public_spd_ceiling_contact_elements,
                                          public_spd_repair_elements,
                                          reset_public_spd_repair_elements, retract_spd_full)

    torch.manual_seed(0)
    Q, _ = torch.linalg.qr(torch.randn(4, 3, 3))
    sigma = Q @ torch.diag_embed(torch.tensor([1.0, 20.0, 200.0]).expand(4, 3)) @ Q.transpose(-1, -2)
    reset_public_spd_repair_elements()
    out = retract_spd_full(sigma, torch.zeros_like(sigma), eps=1e-6, sigma_max=100.0)
    assert public_spd_repair_elements() == 0
    assert public_spd_ceiling_contact_elements() >= 1                       # rows left at sigma_max
    eigenvalues = torch.linalg.eigvalsh(out.double())
    assert float(eigenvalues.max()) < 100.0 and float(eigenvalues.min()) > 0.99

    floor = torch.diag(torch.tensor([1.0, 1.0, 1e-9])).expand(2, 3, 3).clone()
    floor[0] = torch.diag(torch.tensor([1.0, 1.0, 200.0]))                 # one ceiling-only row
    reset_public_spd_repair_elements()
    from vfe3.geometry.retraction import _certify_public_spd
    _certify_public_spd(floor, eps=1e-6, sigma_max=100.0)
    assert public_spd_repair_elements() == 1
    assert public_spd_ceiling_contact_elements() == 1
    reset_public_spd_repair_elements()
    assert public_spd_repair_elements() == 0 and public_spd_ceiling_contact_elements() == 0


# Off-strip Psi inverse gate (hotspot-4#3) ---------------------------------------------------------

def _dense_psi_right_solve(A: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    r"""Solve :math:`\Psi_R(\mathrm{ad}_A)Z = Y` densely, with :math:`\Psi_R(L)` from an augmented exponential."""
    d = A.shape[-1]
    eye = torch.eye(d, dtype=F64)
    ad = torch.kron(A, eye) - torch.kron(eye, A.T.contiguous())                       # row-major vec(AZ - ZA)
    n = d * d
    augmented = torch.zeros(2 * n, 2 * n, dtype=F64)
    augmented[:n, :n] = ad
    augmented[:n, n:] = torch.eye(n, dtype=F64)
    psi = torch.linalg.matrix_exp(augmented)[:n, n:]                       # int_0^1 e^{s ad} ds
    return torch.linalg.solve(psi, Y.reshape(-1)).reshape(d, d)


def _offstrip_block(delta: float) -> torch.Tensor:
    torch.manual_seed(3)
    Q, _ = torch.linalg.qr(torch.randn(4, 4, dtype=F64))
    core = torch.zeros(4, 4, dtype=F64)
    core[:2, :2] = torch.tensor([[0.05, -3.0], [3.0, 0.05]], dtype=F64)      # |Im lambda| = 3 > pi - 0.25
    core[2:, 2:] = torch.tensor([[0.1, 1.0], [delta, 0.1]], dtype=F64)      # near-defective for small delta
    return Q @ core @ Q.T


def test_offstrip_psi_inverse_rejects_rows_outside_the_spectral_budget() -> None:
    from vfe3.geometry.phi_preconditioner import _SPECTRAL_PSI_COND_MAX, _spectral_psi_inverse

    Y = torch.randn(4, 4, dtype=F64, generator=torch.Generator().manual_seed(5))
    good, bad = _offstrip_block(0.5), _offstrip_block(1e-8)
    Z, ok = _spectral_psi_inverse(torch.stack([good, bad]), torch.stack([Y, Y]), left=False)
    assert bool(ok[0]) and not bool(ok[1])
    reference = _dense_psi_right_solve(good, Y)
    assert float((Z[0] - reference).norm() / reference.norm()) <= 1e-10
    assert torch.equal(Z[1], torch.zeros_like(Z[1]))
    _, V = torch.linalg.eig(bad)
    kappa = float(torch.linalg.matrix_norm(V) * torch.linalg.matrix_norm(torch.linalg.inv(V)))
    assert kappa < _SPECTRAL_PSI_COND_MAX                                  # the old gate admitted it


# Generator-free 'l0' block under pullback_group (P2-23, hotspot-4#1) -------------------------------

def test_per_block_pullback_skips_a_generator_free_l0_block_exactly() -> None:
    from vfe3.geometry.groups import get_group
    from vfe3.geometry.phi_preconditioner import _pullback_group_direction_per_block

    group = get_group("so_n")(7, group_n=3, irrep_spec=[("l0", 1), ("l1", 2)], tying="per_head",
                               dtype=F64)
    G = group.generators
    gen = torch.Generator().manual_seed(11)
    phi = 0.3 * torch.randn(4, G.shape[0], generator=gen, dtype=F64)
    grad = torch.randn(4, G.shape[0], generator=gen, dtype=F64)
    blocked = _pullback_group_direction_per_block(grad, phi, G, irrep_dims=[1, 3, 3])
    whole = _pullback_group_direction_per_block(grad, phi, G, irrep_dims=[7])
    assert blocked.current_element is None                                 # stager rebuilds the dense exp
    assert torch.allclose(blocked.v_phi, whole.v_phi, rtol=1e-10, atol=1e-12)
    assert torch.allclose(blocked.xi, whole.xi, rtol=1e-10, atol=1e-12)


def test_untied_so_n_with_l0_block_takes_a_pullback_group_train_step() -> None:
    from vfe3.model.model import VFEModel
    from vfe3.train import build_optimizer, train_step

    cfg = _cfg(vocab_size=12, embed_dim=7, n_heads=3, max_seq_len=8, n_layers=1, pos_phi="none",
               gauge_group="so_n", group_n=3, irrep_spec=[("l0", 1), ("l1", 2)], gauge_tying="per_head",
               m_phi_update_mode="pullback_group", m_phi_group_retraction="chart",
               transport_chart_max_norm=6.0, m_phi_lr=0.1, pos_rotation="none")
    torch.manual_seed(0)
    model = VFEModel(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    before = model.prior_bank.phi_embed.detach().clone()
    loss = train_step(model, optimizer, scheduler, torch.randint(0, 12, (4, 8)),
                      torch.randint(0, 12, (4, 8)), grad_clip=1.0)
    assert math.isfinite(float(loss))
    assert not torch.equal(model.prior_bank.phi_embed.detach(), before)


# fisher_group chart route near the exp singularity (P2-24, geometer#1) -----------------------------

@pytest.mark.parametrize("theta", [3.1, 3.13, 3.2])
def test_offstrip_chart_failure_rejects_the_row_instead_of_killing_the_step(theta: float) -> None:
    from tests.test_phi_fisher import _stub_provider
    from vfe3.gauge_optim import GaugeManifoldAdamW
    from vfe3.geometry.groups import get_group

    group = get_group("block_glk")(6, 2)
    gen = torch.Generator().manual_seed(121)
    phi = nn.Parameter((0.3 * torch.randn(5, 18, generator=gen)).float())
    with torch.no_grad():                                               # row 4, head 0: rotation by theta
        phi[4].zero_()
        phi[4, 1], phi[4, 3] = -theta, theta
    mu = torch.randn(5, 2, 3, generator=gen, dtype=F64)
    s = torch.exp(0.3 * torch.randn(5, 2, 3, generator=gen, dtype=F64))
    counts = torch.tensor([2.0, 0.0, 3.0, 0.0, 1.0], dtype=F64)
    optimizer = GaugeManifoldAdamW(
        [{"params": [phi], "lr": 0.05, "pullback_group": True, "weight_decay": 0.0}],
        group, phi_group_trust_radius=0.1, phi_chart_max_norm=5.0, phi_bch_residual_max=1e-6,
        phi_group_retraction="chart",
        phi_statistics_provider=_stub_provider(mu, s, counts), weight_decay=0.0)
    before = phi.detach().clone()
    phi.grad = torch.zeros_like(phi)
    phi.grad[[0, 2, 4]] = torch.randn(3, 18, generator=gen).float()

    optimizer.step()

    after = phi.detach()
    assert torch.isfinite(after).all()
    assert torch.equal(after[[1, 3]], before[[1, 3]])
    for r in (0, 2):
        assert not torch.equal(after[r], before[r])
    # The pre-fix stager raised FloatingPointError here (residuals 2.2e-6, 3.0e-5, 1.1e-6 > 1e-6).
    assert torch.equal(after[4], before[4]) and optimizer.phi_rejected_rows == 1


# Lazy generator basis in manifold staging (P2-22, hotspot-4#5) -------------------------------------

def test_compact_staging_never_materializes_the_elementary_basis() -> None:
    from vfe3.gauge_optim import (GaugeManifoldAdamW, _pullback_group_current_element,
                                  _pullback_group_product_residual)
    from vfe3.geometry.groups import _ElementaryBlockBasis, get_group

    group = get_group("block_glk")(6, 2)
    assert isinstance(group._generator_source, _ElementaryBlockBasis)
    phi = nn.Parameter(0.1 * torch.randn(3, 18))
    GaugeManifoldAdamW([{"params": [phi], "lr": 0.05, "pullback_group": True, "weight_decay": 0.0}],
                       group, phi_group_trust_radius=0.1, phi_chart_max_norm=5.0,
                       phi_bch_residual_max=1e-6, weight_decay=0.0)
    phi64 = phi.detach().double()
    delta = 1e-3 * torch.randn_like(phi64)
    current = _pullback_group_current_element(phi64, group)
    residual = _pullback_group_product_residual(phi64 + delta, current, delta, group)
    assert isinstance(group._generator_source, _ElementaryBlockBasis)
    assert current.shape == (3, 2, 3, 3) and torch.isfinite(residual).all()
    dense = torch.linalg.matrix_exp(torch.einsum("ra,aij->rij", phi64, group.generators.double()))
    assert torch.allclose(dense[:, :3, :3], current[:, 0], atol=1e-12)


# Map invariants under comparison RoPE (transformer-ml#5) ------------------------------------------

@pytest.mark.parametrize("space", ["comparison", "frame"])
def test_conserved_s_drift_is_withheld_under_comparison_rope(space: str) -> None:
    from tests.test_audit_20261007_opus_fixes import _rope_cfg
    from vfe3.model.model import VFEModel
    from vfe3.posthoc import maps as mp

    cfg = _rope_cfg(space)
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(cfg)
    rows = mp.map_invariants(model.state_dict(), cfg)
    assert rows
    for row in rows:
        if space == "comparison":
            assert math.isnan(row["conserved_S_drift"]) and row["comparison_rope"] == 1.0
            assert math.isfinite(row["scale_rms"])
        else:
            assert math.isfinite(row["conserved_S_drift"]) and "comparison_rope" not in row


# so_n Gram certificate docstring (philosophy-of-science#9) ----------------------------------------

def test_so4_tower_with_one_chiral_block_runs_without_the_gram_certificate() -> None:
    from vfe3.geometry.groups import get_group

    lone = get_group("so_n")(3, group_n=4, irrep_spec=[("wp2", 1)])
    G = lone.generators.double()
    gram = torch.einsum("aij,bij->ab", G, G)
    assert float((gram - torch.diag(gram.diagonal())).abs().max()) > 1e-6
    assert int(torch.linalg.matrix_rank(gram)) == 3
    assert not lone.frobenius_gram_diagonal
    paired = get_group("so_n")(6, group_n=4, irrep_spec=[("wp2", 1), ("wm2", 1)])
    assert paired.frobenius_gram_diagonal
