r"""Numerical Clebsch-Gordan intertwiners over the irrep registry."""

import pytest
import torch

from vfe3.geometry.generators import generate_son
from vfe3.geometry.irreps import irrep_generators


def test_irrep_generators_public_builder():
    G_def = generate_son(3, dtype=torch.float64)
    rho = irrep_generators(G_def, algebra="so", label="l2")
    assert rho.shape == (3, 5, 5)
    assert (rho + rho.transpose(-1, -2)).abs().max() < 1e-12


def test_so3_selection_rules():
    from vfe3.geometry.cg import cg_intertwiners
    # l1 (x) l1 = l0 (+) l1 (+) l2 : each target multiplicity 1
    for c, n in (("l0", 1), ("l1", 1), ("l2", 1), ("l3", 0)):
        C = cg_intertwiners(3, algebra="so", label_a="l1", label_b="l1", label_c=c)
        assert C.shape[0] == n, (c, C.shape)
    # l1 (x) l2 = l1 (+) l2 (+) l3 : no l0
    assert cg_intertwiners(3, algebra="so", label_a="l1", label_b="l2", label_c="l0").shape[0] == 0
    assert cg_intertwiners(3, algebra="so", label_a="l1", label_b="l2", label_c="l3").shape[0] == 1


def test_cg_intertwiner_is_equivariant():
    from vfe3.geometry.cg import cg_intertwiners
    from vfe3.geometry.irreps import irrep_generators
    G_def = generate_son(3, dtype=torch.float64)
    ra = irrep_generators(G_def, algebra="so", label="l1")
    rb = irrep_generators(G_def, algebra="so", label="l2")
    rc = irrep_generators(G_def, algebra="so", label="l2")
    C = cg_intertwiners(3, algebra="so", label_a="l1", label_b="l2", label_c="l2")[0]  # (5, 15)
    gen = torch.Generator().manual_seed(0)
    coeff = 0.4 * torch.randn(3, generator=gen, dtype=torch.float64)
    ga = torch.linalg.matrix_exp(torch.einsum("a,aij->ij", coeff, ra))
    gb = torch.linalg.matrix_exp(torch.einsum("a,aij->ij", coeff, rb))
    gc = torch.linalg.matrix_exp(torch.einsum("a,aij->ij", coeff, rc))
    x = torch.randn(3, generator=gen, dtype=torch.float64)
    y = torch.randn(5, generator=gen, dtype=torch.float64)
    lhs = C @ torch.kron(ga @ x, gb @ y)                 # C(g x (x) g y)
    rhs = gc @ (C @ torch.kron(x, y))                    # g C(x (x) y)
    assert (lhs - rhs).abs().max() < 1e-10


def test_cg_selection_enumerates_admissible_triples():
    from vfe3.geometry.cg import cg_selection
    sel = {(a, b, c) for a, b, c, _ in cg_selection(3, algebra="so",
                                                    labels=["l0", "l1", "l2"])}
    assert ("l1", "l1", "l2") in sel
    assert ("l1", "l2", "l1") in sel
    assert ("l0", "l1", "l1") in sel                     # l0 source acts as a learned gate
    assert ("l1", "l1", "l3") not in sel                 # target not in the spec
    # unordered source pairs: (l2, l1) never appears (canonical order a <= b)
    assert all(a <= b for a, b, _c in sel)


def test_cg_cost_guard():
    from vfe3.geometry.cg import cg_intertwiners
    with pytest.raises(ValueError, match="construction size"):
        cg_intertwiners(8, algebra="so", label_a="l3", label_b="l3", label_c="l3")


def test_cg_cache_immune_to_caller_mutation():
    from vfe3.geometry.cg import cg_intertwiners
    C1 = cg_intertwiners(3, algebra="so", label_a="l1", label_b="l1", label_c="l2")
    C1.mul_(0.0)                                         # caller misbehaves in place
    C2 = cg_intertwiners(3, algebra="so", label_a="l1", label_b="l1", label_c="l2")
    assert C2.abs().max() > 0                            # cache unharmed
    assert C1.data_ptr() != C2.data_ptr()                # no aliasing


def _tower_group():
    from vfe3.geometry.groups import get_group
    return get_group("so_n")(9, group_n=3,
                             irrep_spec=[("l0", 1), ("l1", 1), ("l2", 1)],
                             dtype=torch.float64)


def test_cg_coupling_zero_init_is_exact_passthrough():
    from vfe3.model.cg_coupling import CGCoupling
    grp = _tower_group()
    cpl = CGCoupling(3, "so", grp.irrep_dims, grp.irrep_labels).double()
    assert cpl.path_weights.shape[0] > 0
    mu = torch.randn(2, 4, 9, dtype=torch.float64)
    sig = torch.rand(2, 4, 9, dtype=torch.float64)
    mu2, sig2 = cpl(mu, sig)
    assert torch.equal(mu2, mu) and torch.equal(sig2, sig)


def test_cg_coupling_means_update_is_exactly_equivariant():
    from vfe3.model.cg_coupling import CGCoupling
    grp = _tower_group()
    cpl = CGCoupling(3, "so", grp.irrep_dims, grp.irrep_labels).double()
    with torch.no_grad():
        cpl.path_weights.copy_(0.3 * torch.randn(cpl.path_weights.shape[0],
                                                 dtype=torch.float64))
    g = torch.linalg.matrix_exp(
        torch.einsum("a,aij->ij", 0.4 * torch.randn(3, dtype=torch.float64), grp.generators))
    mu = torch.randn(5, 9, dtype=torch.float64)
    sig = torch.rand(5, 9, dtype=torch.float64)
    out_then_g = torch.einsum("kl,nl->nk", g, cpl(mu, sig)[0])
    g_then_out = cpl(torch.einsum("kl,nl->nk", g, mu), sig)[0]
    assert (out_then_g - g_then_out).abs().max() < 1e-12


def test_cg_coupling_self_product_reaches_other_types():
    # zero everything except one l1 (x) l1 -> l2 path: the l2 head must move, others must not.
    from vfe3.model.cg_coupling import CGCoupling
    grp = _tower_group()
    cpl = CGCoupling(3, "so", grp.irrep_dims, grp.irrep_labels).double()
    idx = next(p for p, (a, b, c) in enumerate(cpl.path_types)
               if (a, b, c) == ("l1", "l1", "l2"))
    with torch.no_grad():
        cpl.path_weights[idx] = 1.0
    mu = torch.randn(3, 9, dtype=torch.float64)
    mu2, _ = cpl(mu, torch.ones(3, 9, dtype=torch.float64))
    assert not torch.allclose(mu2[:, 4:9], mu[:, 4:9])   # l2 head updated
    assert torch.equal(mu2[:, 0:4], mu[:, 0:4])          # l0 and l1 heads untouched


def test_cg_coupling_full_cov_sigma_passes_through():
    from vfe3.model.cg_coupling import CGCoupling
    grp = _tower_group()
    cpl = CGCoupling(3, "so", grp.irrep_dims, grp.irrep_labels).double()
    with torch.no_grad():
        cpl.path_weights.fill_(0.3)
    mu = torch.randn(2, 4, 9, dtype=torch.float64)
    S = torch.randn(2, 4, 9, 9, dtype=torch.float64)
    mu2, S2 = cpl(mu, S)
    assert S2 is S                                       # untouched, same object
    assert not torch.equal(mu2, mu)                      # means did move


# ---------------------------------------------------------------------------
# PB-13 (Task 4, 2026-07-12): analytic CG moment closure -- Jacobian, delta-method
# covariance, and the post-CG moment divergence rows.
# ---------------------------------------------------------------------------

def _l2_group():
    from vfe3.geometry.groups import get_group
    # single l2 block (dim 5): l2 (x) l2 -> l2 is a symmetric slot, so the self-pair survives the
    # prune (one live path). K = 5 exercises the Jacobian/covariance on a small CPU-bound tower.
    return get_group("so_n")(5, group_n=3, irrep_spec=[("l2", 1)], dtype=torch.float64)


def _l2_coupling(mode="passthrough", weight=None):
    from vfe3.model.cg_coupling import CGCoupling
    grp = _l2_group()
    cpl = CGCoupling(3, "so", grp.irrep_dims, grp.irrep_labels,
                     cg_covariance_mode=mode).double()
    if weight is not None:
        with torch.no_grad():
            cpl.path_weights.copy_(weight * torch.randn(cpl.path_weights.shape[0],
                                                        dtype=torch.float64))
    return cpl, grp


def _sym_spd(K):
    M = torch.randn(K, K, dtype=torch.float64)
    S = M @ M.transpose(-1, -2)
    S = 0.5 * (S + S.transpose(-1, -2))                                       # bit-exactly symmetric
    return S + K * torch.eye(K, dtype=torch.float64)                          # ...and SPD (PSD + K I)


def test_cg_forward_moments_jacobian_matches_autograd():
    cpl, _ = _l2_coupling(weight=0.5)
    sigma = _sym_spd(5)
    mu = torch.randn(5, dtype=torch.float64)
    res = cpl.forward_moments(mu, sigma)
    assert res.jacobian.shape == (5, 5)

    def f(m):
        return cpl.forward_moments(m, sigma).mu

    J_auto = torch.autograd.functional.jacobian(f, mu)
    assert torch.allclose(res.jacobian, J_auto, atol=1e-10, rtol=0.0)


def test_cg_forward_moments_jacobian_matches_autograd_multipath():
    r"""Multi-path [l0, l1] tower (K=4): three live paths -- the (l0,l0,l0) and (l1,l1,l0)
    self-pairs plus the (l0,l1,l1) distinct-source pair -- so the OFF-diagonal (target != source)
    Jacobian blocks and the self-pair double contraction are all exercised, not just the single
    l2 self-pair block above."""
    from vfe3.model.cg_coupling import CGCoupling
    from vfe3.geometry.groups import get_group
    grp = get_group("so_n")(4, group_n=3, irrep_spec=[("l0", 1), ("l1", 1)],
                            dtype=torch.float64)
    cpl = CGCoupling(3, "so", grp.irrep_dims, grp.irrep_labels).double()
    assert len(cpl.paths) >= 3                                   # multi-path, multi-group tower
    torch.manual_seed(2)
    with torch.no_grad():
        cpl.path_weights.copy_(0.7 * torch.randn(cpl.path_weights.shape[0],
                                                 dtype=torch.float64))
    mu = torch.randn(4, dtype=torch.float64)
    sigma = torch.rand(4, dtype=torch.float64) + 0.1

    def f(m):
        return cpl.forward_moments(m, sigma).mu

    res = cpl.forward_moments(mu, sigma)
    J_auto = torch.autograd.functional.jacobian(f, mu)
    assert torch.allclose(res.jacobian, J_auto, atol=1e-10, rtol=0.0)


def test_cg_forward_moments_delta_covariance_is_J_sigma_JT_plus_covariant_floor_spd():
    cpl, _ = _l2_coupling(mode="delta_full", weight=0.5)
    sigma = _sym_spd(5)
    mu = torch.randn(5, dtype=torch.float64)
    res = cpl.forward_moments(mu, sigma)
    J = res.jacobian
    pushed = J @ sigma @ J.transpose(-1, -2)
    expected = 0.5 * (pushed + pushed.transpose(-1, -2)) \
        + cpl.cg_covariance_floor * sigma
    assert torch.allclose(res.sigma, expected, atol=1e-12, rtol=0.0)
    assert torch.allclose(res.sigma, res.sigma.transpose(-1, -2), atol=1e-14)   # symmetric
    assert (torch.linalg.eigvalsh(res.sigma) > 0).all()                          # SPD


def test_cg_forward_moments_zero_weight_preserves_covariance_with_documented_floor():
    cpl, _ = _l2_coupling(mode="delta_full")           # path_weights zero-init
    sigma = _sym_spd(5)
    mu = torch.randn(5, dtype=torch.float64)
    res = cpl.forward_moments(mu, sigma)
    assert torch.equal(res.mu, mu)                                               # mu_out = mu exactly
    expected_sigma = (1.0 + cpl.cg_covariance_floor) * sigma
    assert torch.allclose(res.sigma, expected_sigma, atol=1e-12, rtol=0.0)
    assert torch.equal(res.jacobian, torch.eye(5, dtype=torch.float64))          # J = I exactly


def test_cg_forward_moments_covariance_is_gauge_congruent():
    cpl, grp = _l2_coupling(mode="delta_full", weight=0.4)
    g = torch.linalg.matrix_exp(
        torch.einsum("a,aij->ij", 0.3 * torch.randn(3, dtype=torch.float64), grp.generators))
    mu = torch.randn(5, dtype=torch.float64)
    sigma = _sym_spd(5)
    res = cpl.forward_moments(mu, sigma)
    res_g = cpl.forward_moments(torch.einsum("kl,l->k", g, mu),
                                torch.einsum("ik,kl,jl->ij", g, sigma, g))
    assert torch.allclose(res_g.mu, torch.einsum("kl,l->k", g, res.mu), atol=1e-10)
    assert torch.allclose(res_g.sigma,
                          torch.einsum("ik,kl,jl->ij", g, res.sigma, g), atol=1e-9)


def test_cg_forward_selects_passthrough_vs_delta_full():
    cpl_pass, _ = _l2_coupling(mode="passthrough", weight=0.5)
    cpl_full, _ = _l2_coupling(mode="delta_full", weight=0.5)
    with torch.no_grad():                                # identical weights across the two modes
        cpl_full.path_weights.copy_(cpl_pass.path_weights)
    mu = torch.randn(3, 5, dtype=torch.float64)
    sigma = _sym_spd(5).expand(3, 5, 5).contiguous()
    mu_p, sig_p = cpl_pass(mu, sigma)
    mu_f, sig_f = cpl_full(mu, sigma)
    assert torch.equal(mu_p, mu_f)                       # the mean update is mode-independent
    assert sig_p is sigma                                # passthrough returns sigma untouched
    assert not torch.allclose(sig_f, sigma)             # delta_full pushes covariance forward


def _e2e_cfg(**kw):
    from vfe3.config import VFE3Config
    base = dict(vocab_size=20, embed_dim=9, n_heads=3, max_seq_len=5, n_layers=1,
                n_e_steps=1, e_q_mu_lr=0.05, e_phi_lr=0.0,
                gauge_group="so_n", group_n=3,
                irrep_spec=[("l0", 1), ("l1", 1), ("l2", 1)],
                phi_precond_mode="none")
    base.update(kw)
    return VFE3Config(**base)


def test_use_cg_coupling_rejected_off_towers():
    from vfe3.config import VFE3Config
    with pytest.raises(ValueError, match="use_cg_coupling"):
        VFE3Config(vocab_size=8, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=1,
                   gauge_group="block_glk", use_cg_coupling=True)


def test_cg_model_step0_byte_identical_and_trains():
    from vfe3.model.model import VFEModel
    from vfe3.train import build_optimizer
    tok = torch.randint(0, 20, (2, 5)); tgt = torch.randint(0, 20, (2, 5))
    torch.manual_seed(0)
    base = VFEModel(_e2e_cfg())
    torch.manual_seed(0)
    cg = VFEModel(_e2e_cfg(use_cg_coupling=True))
    lg_base = base(tok); lg_cg = cg(tok)
    assert torch.equal(lg_base, lg_cg)                  # zero-init: step 0 byte-identical
    build_optimizer(cg, cg.cfg)                         # exact-coverage guard must pass
    with torch.no_grad():
        cg.cg_coupling.path_weights.add_(0.05)
    _, loss, _ = cg(tok, tgt)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(cg.cg_coupling.path_weights.grad).all()
    assert cg.cg_coupling.path_weights.grad.abs().sum() > 0
