import math

import pytest
import torch

import vfe3.metrics as metrics_module
from vfe3.geometry.groups import get_group
from vfe3.geometry.transport import CompactFactoredTransport, compute_transport_operators
from vfe3.metrics import (
    attention_distance_decay,
    attention_entropy,
    attention_entropy_rows,
    belief_spectrum,
    bootstrap_ce_band,
    bootstrap_token_ce_band,
    causal_sanity,
    compute_metrics,
    curvature_field,
    effective_rank,
    effective_rank_per_token,
    energy_directedness,
    estep_residuals,
    fisher_trace,
    free_energy_full_decomposition,
    free_energy_terms,
    gauge_equivariance_residual,
    gauge_trace_spread,
    group_gauge_invariant,
    guard_saturation,
    head_redundancy_js,
    holonomy_deviation,
    holonomy_deviation_sampled,
    holonomy_wilson_sampled,
    per_head_gauge_invariants,
    positional_content_score,
    self_coupling_profile,
    spearman_rho,
    spd_geodesic_distance,
    structured_head_scores,
    transport_asymmetry,
)


def test_effective_rank_flat_and_peaked():
    assert torch.allclose(effective_rank(torch.ones(4)), torch.tensor(4.0), atol=1e-5)
    assert torch.allclose(effective_rank(torch.ones(2)), torch.tensor(2.0), atol=1e-5)
    peaked = torch.tensor([1.0, 1e-9, 1e-9, 1e-9])
    assert abs(float(effective_rank(peaked)) - 1.0) < 1e-3        # one dominant mode -> ~1


def test_attention_entropy_uniform_and_onehot():
    N = 5
    uniform = torch.full((3, N, N), 1.0 / N)
    assert abs(float(attention_entropy(uniform)) - math.log(N)) < 1e-4
    onehot = torch.zeros(2, N, N); onehot[..., 0] = 1.0
    assert float(attention_entropy(onehot)) < 1e-5


def test_free_energy_terms_is_registered_metric():
    """free_energy_terms is selectable through the metrics registry (compute_metrics), not only
    as a bare function."""
    N = 4
    self_div = torch.rand(N)
    energy = torch.rand(N, N)
    beta = torch.softmax(-energy, dim=-1)
    alpha = torch.ones(N)
    out = compute_metrics(["free_energy_terms"], self_div=self_div, energy=energy,
                          beta=beta, alpha=alpha, tau=1.0)
    assert "free_energy_terms" in out and "total" in out["free_energy_terms"]


def test_holonomy_deviation_zero_for_flat_cocycle():
    grp = get_group("glk")(3)
    phi = 0.2 * torch.randn(1, 4, grp.generators.shape[0])
    omega = compute_transport_operators(phi, grp)["Omega"][0]      # (4,4,3,3) flat cocycle
    assert float(holonomy_deviation(omega)) < 1e-4                 # every triangle closes (H=I)


def test_holonomy_deviation_positive_for_non_cocycle():
    g = torch.Generator().manual_seed(0)
    N, K = 4, 3
    omega = torch.eye(K).expand(N, N, K, K) + 0.3 * torch.randn(N, N, K, K, generator=g)
    assert float(holonomy_deviation(omega)) > 1e-2                 # random transport does not close


def test_gauge_trace_spread_zero_at_phi_zero():
    grp = get_group("glk")(3)
    G = grp.generators
    assert float(gauge_trace_spread(torch.zeros(5, G.shape[0]), G)) < 1e-7
    assert float(gauge_trace_spread(torch.randn(5, G.shape[0]), G)) > 0.0


def test_free_energy_terms_decomposition():
    N = 3
    self_div = torch.zeros(N)                                      # q == p -> self term 0
    energy = torch.rand(N, N)
    beta = torch.softmax(-energy, dim=-1)
    alpha = torch.ones(N)
    terms = free_energy_terms(self_div, energy, beta, alpha, tau=1.0)
    assert abs(terms["self_coupling"]) < 1e-6
    assert abs(terms["total"] - (terms["self_coupling"] + terms["belief_coupling"] + terms["attention_entropy"])) < 1e-5


def test_compute_metrics_registry_record():
    grp = get_group("glk")(3)
    phi = 0.1 * torch.randn(1, 4, grp.generators.shape[0])
    omega = compute_transport_operators(phi, grp)["Omega"][0]
    rec = compute_metrics(
        ["effective_rank", "attention_entropy", "holonomy_deviation", "gauge_trace_spread"],
        sigma=torch.rand(4, 3) + 0.5,
        diagonal=True,                                                # effective_rank REQUIRES the flag (PB-07)
        beta=torch.softmax(torch.randn(4, 4), dim=-1),
        omega=omega,
        phi=phi[0],
        generators=grp.generators,
    )
    assert set(rec) == {"effective_rank", "attention_entropy", "holonomy_deviation", "gauge_trace_spread"}
    assert all(isinstance(v, float) for v in rec.values())


# --- publication-figure metrics --------------------------------------------

def test_effective_rank_per_token_keeps_distribution():
    sigma = torch.rand(6, 4) + 0.1
    per = effective_rank_per_token(sigma)
    assert per.shape == (6,)                                       # per-token, not mean-reduced
    assert torch.allclose(per.mean(), effective_rank(sigma).mean(), atol=1e-5)
    per_full = effective_rank_per_token(torch.diag_embed(sigma))   # full-cov path == diagonal
    assert torch.allclose(per_full, per, atol=1e-4)


def test_spd_geodesic_distance_zero_symmetric_diag_full_agree():
    a = torch.rand(5, 3) + 0.2
    b = torch.rand(5, 3) + 0.2
    assert float(spd_geodesic_distance(a, a).abs().max()) < 1e-5   # d(S, S) = 0
    d_ab = spd_geodesic_distance(a, b)
    assert torch.allclose(d_ab, spd_geodesic_distance(b, a), atol=1e-4)   # symmetric
    A, B = torch.diag_embed(a), torch.diag_embed(b)               # full embedding agrees
    assert torch.allclose(spd_geodesic_distance(A, B), d_ab, atol=1e-4)


def test_spd_geodesic_distance_uses_airm_not_half_fisher_scale():
    sigma_a = torch.ones(1, 1)
    sigma_b = torch.exp(torch.tensor([[2.0]]))

    assert spd_geodesic_distance(sigma_a, sigma_b).item() == pytest.approx(2.0)


def test_spd_geodesic_distance_full_uses_cholesky_generalized_eigenvalues(monkeypatch):
    transform = torch.tensor([[2.0, 0.5], [0.0, 0.25]], dtype=torch.float32)
    log_eigenvalues = torch.tensor([1.0, -2.0], dtype=torch.float32)
    sigma_a = transform @ transform.transpose(-1, -2)
    sigma_b = transform @ torch.diag(torch.exp(log_eigenvalues)) @ transform.transpose(-1, -2)

    def _forbid_explicit_eigh(*args, **kwargs):
        raise AssertionError("full SPD distance explicitly eigendecomposed Sigma_a")

    monkeypatch.setattr(torch.linalg, "eigh", _forbid_explicit_eigh)
    distance = spd_geodesic_distance(sigma_a, sigma_b, diagonal=False)

    assert distance.dtype == sigma_a.dtype
    assert torch.allclose(distance, log_eigenvalues.square().sum().sqrt(), atol=2e-5)


def test_belief_spectrum_eigenvalues_and_condition():
    sigma = torch.tensor([[4.0, 1.0, 0.25]])                      # K=3 diagonal
    sp = belief_spectrum(sigma)
    assert torch.allclose(sp["eigenvalues"][0], torch.tensor([4.0, 1.0, 0.25]), atol=1e-5)
    assert abs(float(sp["condition"][0]) - 16.0) < 1e-4           # 4.0 / 0.25
    assert torch.allclose(sp["effective_rank"], effective_rank_per_token(sigma), atol=1e-5)


def test_fisher_trace_diag_full_agree():
    sigma = torch.rand(5, 3) + 0.3
    ft = fisher_trace(sigma)
    assert ft.shape == (5,)
    assert torch.allclose(ft, (0.5 / sigma).sum(-1), atol=1e-5)
    assert torch.allclose(fisher_trace(torch.diag_embed(sigma)), ft, atol=1e-4)
    # C18: keyword-only args follow the convention order (defined float eps BEFORE the Optional
    # diagonal); calling by keyword in that order matches the positional-sigma result.
    assert torch.allclose(fisher_trace(sigma, eps=1e-9, diagonal=True), ft, atol=1e-6)
    assert torch.allclose(fisher_trace(torch.diag_embed(sigma), eps=1e-9, diagonal=False), ft, atol=1e-4)


def test_half_fisher_trace_is_the_named_implementation_and_fisher_trace_alias():
    half_fisher_trace = getattr(metrics_module, "half_fisher_trace", None)
    assert half_fisher_trace is not None
    assert fisher_trace is half_fisher_trace
    assert "UMAP" not in (half_fisher_trace.__doc__ or "")

    sigma = torch.tensor([[2.0, 4.0]])
    expected = 0.5 * torch.linalg.inv(torch.diag_embed(sigma)).diagonal(
        dim1=-2, dim2=-1
    ).sum(-1)
    assert torch.allclose(half_fisher_trace(sigma), expected)
    assert torch.allclose(half_fisher_trace(torch.diag_embed(sigma)), expected)


def test_attention_entropy_rows_reduces_to_global():
    beta = torch.softmax(torch.randn(2, 5, 5), dim=-1)
    rows = attention_entropy_rows(beta)
    assert rows.shape == (2, 5)
    assert torch.allclose(rows.mean(), attention_entropy(beta), atol=1e-5)


def test_causal_sanity_on_causal_softmax():
    N = 6
    logits = torch.randn(N, N)
    mask = torch.tril(torch.ones(N, N, dtype=torch.bool))
    beta = torch.softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)
    out = causal_sanity(beta)
    assert float(out["future_leakage"]) < 1e-6                    # no mass above the diagonal
    assert float(out["row_sum_error"]) < 1e-5                     # rows sum to 1
    assert abs(float(out["active_set_slope"]) - 1.0) < 0.2        # one new key per query


def test_guard_saturation_inert_then_binds():
    sigma = torch.rand(4, 3) + 0.5                                # well inside [eps, sigma_max]
    energy = torch.rand(4, 4) * 5.0                               # well below kl_max
    self_div = torch.rand(4) * 2.0
    inert = guard_saturation(sigma, energy, self_div, eps=1e-6, sigma_max=5.0, kl_max=100.0)
    assert all(v < 1e-9 for v in inert.values())                 # nothing pinned on the pure path
    pinned = guard_saturation(torch.full((4, 3), 5.0), energy, self_div, sigma_max=5.0)
    assert pinned["sigma_ceil_frac"] > 0.99                       # variance ceiling binds


# --- gauge invariants ------------------------------------------------------

def test_group_gauge_invariant_glk_volume_and_zero_at_identity():
    grp = get_group("glk")(3)
    exp0 = compute_transport_operators(torch.zeros(1, 4, grp.generators.shape[0]), grp)["exp_phi"][0]
    assert float(group_gauge_invariant(exp0, grp).abs().max()) < 1e-6      # det I = 1 -> logdet 0
    phi = 0.2 * torch.randn(1, 4, grp.generators.shape[0])
    exp_phi = compute_transport_operators(phi, grp)["exp_phi"][0]
    inv = group_gauge_invariant(exp_phi, grp)
    assert torch.allclose(inv, torch.linalg.slogdet(exp_phi).logabsdet, atol=1e-5)


def test_group_gauge_invariant_so_k_rotation_angle():
    grp = get_group("so_k")(4)
    exp0 = compute_transport_operators(torch.zeros(1, 3, grp.generators.shape[0]), grp)["exp_phi"][0]
    assert float(group_gauge_invariant(exp0, grp).abs().max()) < 1e-6      # no rotation
    phi = 0.3 * torch.randn(1, 3, grp.generators.shape[0])
    exp_phi = compute_transport_operators(phi, grp)["exp_phi"][0]
    assert float(group_gauge_invariant(exp_phi, grp).min()) >= 0.0         # |angles| >= 0


def test_per_head_gauge_invariants_blocks_and_identity():
    grp = get_group("block_glk")(4, 2)
    exp0 = compute_transport_operators(torch.zeros(1, 3, grp.generators.shape[0]), grp)["exp_phi"][0]
    out = per_head_gauge_invariants(exp0, grp.irrep_dims)
    assert out["logdet"].shape == (3, 2) and out["anisotropy"].shape == (3, 2)
    assert float(out["logdet"].abs().max()) < 1e-6                # identity blocks
    # The eigenvalue-modulus squeeze is 0 on an isometric block (all |lambda| == 1), where the old
    # singular-value RATIO was 1. Audit 2026-07-27 replaced the formula; see the invariance test.
    assert torch.allclose(out["anisotropy"], torch.zeros_like(out["anisotropy"]), atol=1e-5)


def test_per_head_gauge_invariants_are_conjugation_invariant():
    r"""Both returned entries must be conjugacy-class functions under the group's OWN gauge action.

    The regression this pins (audit 2026-07-27): ``anisotropy`` was ``s_max / s_min``, invariant
    only under ORTHOGONAL conjugation, while the action here is GL(K) conjugation. It moved by up
    to 301% under an in-group frame change and was published to metrics.csv under a ``gauge_head_*``
    key. This is the per-head twin of the ``sp`` fix already pinned in
    tests/test_run_diagnostics_2026_06_13.py, and its absence is how the defect survived.
    """
    torch.manual_seed(0)
    grp = get_group("block_glk")(4, 2)
    dims = grp.irrep_dims

    def _blockdiag(mats):
        out = torch.zeros(4, 4, dtype=mats[0].dtype)
        start = 0
        for m in mats:
            d = m.shape[-1]
            out[start:start + d, start:start + d] = m
            start += d
        return out

    # A real vertex factor is block-diagonal, and block_glk's gauge is untied per block.
    exp_phi = torch.stack([_blockdiag([torch.matrix_exp(0.3 * torch.randn(d, d)) for d in dims])
                           for _ in range(5)])
    g = _blockdiag([torch.matrix_exp(0.5 * torch.randn(d, d)) for d in dims])
    moved = g @ exp_phi @ torch.linalg.inv(g)

    base, conj = per_head_gauge_invariants(exp_phi, dims), per_head_gauge_invariants(moved, dims)
    for key in ("logdet", "anisotropy"):
        rel = ((conj[key] - base[key]).abs() / base[key].abs().clamp(min=1e-12)).max()
        assert float(rel) < 1e-5, f"{key} moved {float(rel):.3e} under an in-group conjugation"

    # and the old formula would have failed this test, so it is not vacuous
    old = lambda m: (lambda s: s[..., 0] / s[..., -1])(torch.linalg.svdvals(m[..., :dims[0], :dims[0]]))
    assert float(((old(moved) - old(exp_phi)).abs() / old(exp_phi).abs()).max()) > 1e-2


# --- transport / energy directedness --------------------------------------

def test_transport_asymmetry_zero_at_identity_positive_for_gauge():
    grp = get_group("glk")(3)
    omega0 = compute_transport_operators(torch.zeros(1, 5, grp.generators.shape[0]), grp)["Omega"][0]
    assert float(transport_asymmetry(omega0).abs().max()) < 1e-6  # Omega = I -> symmetric
    omega = compute_transport_operators(0.3 * torch.randn(1, 5, grp.generators.shape[0]), grp)["Omega"][0]
    assert float(transport_asymmetry(omega).max()) > 1e-2         # directed transport


def test_energy_directedness_symmetric_vs_asymmetric():
    sym = torch.rand(5, 5); sym = sym + sym.t()
    out = energy_directedness(sym)
    assert float(out["abs_asymmetry"]) < 1e-6
    asym = torch.triu(torch.rand(5, 5) + 0.5, diagonal=1)         # upper-only -> asymmetric
    assert float(energy_directedness(asym)["abs_asymmetry"]) > 1e-2


# --- attention structure ---------------------------------------------------

def test_structured_head_scores_prev_token():
    N = 6
    beta = torch.zeros(2, N, N)
    for i in range(1, N):
        beta[:, i, i - 1] = 1.0
    beta[:, 0, 0] = 1.0
    out = structured_head_scores(beta, period=3)
    assert float(out["prev_token"].min()) > 0.5                   # mass on the previous token
    assert out["prev_token"].shape == (2,)


def test_head_redundancy_js_identical_vs_distinct():
    N = 5
    base = torch.softmax(torch.randn(N, N), dim=-1)
    beta = torch.stack([base, base], dim=0)                       # two identical heads
    js = head_redundancy_js(beta)
    assert float(js[0, 1]) < 1e-6
    other = torch.softmax(torch.randn(N, N), dim=-1)
    js2 = head_redundancy_js(torch.stack([base, other], dim=0))
    assert float(js2[0, 1]) > 1e-3


def test_attention_distance_decay_shape_and_offsets():
    beta = torch.softmax(torch.randn(2, 7, 7), dim=-1)
    out = attention_distance_decay(beta)
    assert out["profile"].shape == (2, 7) and out["offsets"].shape == (7,)


def test_positional_content_score_distance_driven():
    N = 8
    ii = torch.arange(N).unsqueeze(-1); jj = torch.arange(N).unsqueeze(0)
    logits = -(ii - jj).float().masked_fill(ii < jj, float("inf"))
    beta = torch.softmax(logits.masked_fill(ii < jj, float("-inf")), dim=-1)
    assert float(positional_content_score(beta)) > 0.8           # log beta linear in offset


# --- holonomy / curvature --------------------------------------------------

def test_holonomy_deviation_sampled_flat_vs_random():
    grp = get_group("glk")(3)
    omega = compute_transport_operators(0.2 * torch.randn(1, 8, grp.generators.shape[0]), grp)["Omega"][0]
    flat = holonomy_deviation_sampled(omega, n_triples=64, seed=0)
    assert float(flat["mean"]) < 1e-3                            # flat cocycle closes
    g = torch.Generator().manual_seed(0)
    rand = torch.eye(3).expand(8, 8, 3, 3) + 0.3 * torch.randn(8, 8, 3, 3, generator=g)
    assert float(holonomy_deviation_sampled(rand, n_triples=64, seed=0)["mean"]) > 1e-2


def test_curvature_field_flat_is_zero():
    grp = get_group("glk")(3)
    omega = compute_transport_operators(0.2 * torch.randn(1, 6, grp.generators.shape[0]), grp)["Omega"][0]
    assert float(curvature_field(omega, anchor=0).abs().max()) < 1e-3


def test_compact_curvature_field_flat_backward_is_finite():
    exp_blocks = torch.eye(2).reshape(1, 1, 2, 2).expand(3, 1, 2, 2).clone().requires_grad_()
    inv_blocks = torch.eye(2).reshape(1, 1, 2, 2).expand(3, 1, 2, 2).clone().requires_grad_()
    compact = CompactFactoredTransport(exp_blocks, inv_blocks, K=2)

    curvature_field(compact).sum().backward()

    assert torch.isfinite(exp_blocks.grad).all()
    assert torch.isfinite(inv_blocks.grad).all()


def test_holonomy_wilson_flat_cocycle_is_unity():
    # Wilson observable W/K = Re Tr(H)/K -> 1 (deviation 1 - W/K -> 0) when every triangle closes.
    grp = get_group("glk")(3)
    omega = compute_transport_operators(0.2 * torch.randn(1, 8, grp.generators.shape[0]), grp)["Omega"][0]
    out = holonomy_wilson_sampled(omega, n_triples=64, seed=0)
    assert abs(float(out["wilson_mean"]) - 1.0) < 1e-3
    assert float(out["deviation_mean"]) < 1e-3


def test_holonomy_wilson_constant_rotation_matches_analytic():
    # omega_ij = R(theta) for all distinct pairs -> H_ijk = R(theta)^3 = R(3*theta),
    # so Re Tr(H)/K = Tr(R(3*theta))/2 = cos(3*theta) for every triple.
    theta = 0.3
    c, s = math.cos(theta), math.sin(theta)
    R = torch.tensor([[c, -s], [s, c]])
    omega = R.expand(5, 5, 2, 2).contiguous()
    out = holonomy_wilson_sampled(omega, n_triples=32, seed=1)
    assert abs(float(out["wilson_mean"]) - math.cos(3.0 * theta)) < 1e-4
    assert torch.allclose(out["per_triple"], torch.full_like(out["per_triple"], math.cos(3.0 * theta)), atol=1e-4)


def test_holonomy_wilson_per_head_decomposition_averages_to_full():
    # Tr(H) = sum_h Tr(H^(h)) over the n_heads diagonal blocks, so the per-head normalized
    # Wilson values (each Tr(H^(h))/d_k) average back to the full Wilson observable Tr(H)/K.
    g = torch.Generator().manual_seed(0)
    omega = torch.eye(4).expand(6, 6, 4, 4) + 0.3 * torch.randn(6, 6, 4, 4, generator=g)
    out = holonomy_wilson_sampled(omega, n_heads=2, n_triples=64, seed=0)
    assert out["per_head"].shape == (2,)
    assert abs(float(out["per_head"].mean()) - float(out["wilson_mean"])) < 1e-5


def test_wilson_trace_is_conjugation_invariant():
    """C6: the Wilson deviation 1 - Re Tr(H)/K is invariant under a per-vertex NON-orthogonal
    GL(K) conjugation Omega_ij -> g_i Omega_ij g_j^{-1} (the holonomy conjugates, H -> g_i H
    g_i^{-1}, and Tr is a class function), while the Frobenius ||H - I||_F deviation is
    frame-dependent and CHANGES -- the invariant/frame-dependent split the labels now assert."""
    g_rng = torch.Generator().manual_seed(0)
    N, K = 6, 3
    omega = (torch.eye(K, dtype=torch.float64).expand(N, N, K, K)
             + 0.3 * torch.randn(N, N, K, K, generator=g_rng, dtype=torch.float64))
    # invertible, deliberately NON-orthogonal per-vertex frames g_i = I + small random
    g_i = (torch.eye(K, dtype=torch.float64).expand(N, K, K)
           + 0.3 * torch.randn(N, K, K, generator=g_rng, dtype=torch.float64))
    g_inv = torch.linalg.inv(g_i)
    omega_conj = torch.einsum("ikl,ijlm,jmn->ijkn", g_i, omega, g_inv)

    wil = holonomy_wilson_sampled(omega, n_triples=64, seed=0)
    wil_c = holonomy_wilson_sampled(omega_conj, n_triples=64, seed=0)
    assert abs(float(wil["deviation_mean"]) - float(wil_c["deviation_mean"])) < 1e-5

    fro = holonomy_deviation_sampled(omega, n_triples=64, seed=0)
    fro_c = holonomy_deviation_sampled(omega_conj, n_triples=64, seed=0)
    assert abs(float(fro["mean"]) - float(fro_c["mean"])) > 1e-3


def test_holonomy_frobenius_and_wilson_both_flat_zero():
    """C6: both the frame-dependent Frobenius deviation and the gauge-invariant Wilson deviation
    are ~0 on the flat cocycle -- the labeling split changes reporting, not the flatness oracle."""
    grp = get_group("glk")(3)
    omega = compute_transport_operators(0.2 * torch.randn(1, 8, grp.generators.shape[0]), grp)["Omega"][0]
    assert float(holonomy_deviation_sampled(omega, n_triples=64, seed=0)["mean"]) < 1e-3
    assert float(holonomy_wilson_sampled(omega, n_triples=64, seed=0)["deviation_mean"]) < 1e-3


def test_holonomy_wilson_rejects_indivisible_head_count():
    omega = torch.eye(3).expand(4, 4, 3, 3).contiguous()
    try:
        holonomy_wilson_sampled(omega, n_heads=2)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when n_heads does not divide K")


# --- free-energy closure / profile / residuals -----------------------------

def test_free_energy_full_decomposition_closes():
    out = free_energy_full_decomposition(2.0, 10.0, 5.0, 188.0, lambda_beta=2.0)
    assert abs(out["total"] - (2.0 + 2.0 * 10.0 + 2.0 * 5.0 + 188.0)) < 1e-6
    assert abs(out["belief_coupling"] - 20.0) < 1e-6                # lambda_beta scaling applied


def test_self_coupling_profile_per_coord_sums():
    self_div = torch.rand(5, 3)                                    # per-coordinate
    alpha = torch.ones(5, 3)
    prof = self_coupling_profile(self_div, alpha)
    assert prof["self_div"].shape == (5,)
    assert torch.allclose(prof["self_coupling_per_token"], self_div.sum(-1), atol=1e-5)


def test_estep_residuals_zero_for_constant_trajectory():
    mu = torch.randn(4, 6, 3).cumsum(0)                           # moving means
    const_mu = mu[:1].expand(4, 6, 3)                             # constant -> zero residual
    sigma = (torch.rand(4, 6, 3) + 0.5)
    const_sig = sigma[:1].expand(4, 6, 3)
    phi = torch.randn(4, 6, 2)
    const_phi = phi[:1].expand(4, 6, 2)
    res = estep_residuals(const_mu, const_sig, const_phi)
    assert float(res["r_mu"].abs().max()) < 1e-6
    assert float(res["r_sigma"].abs().max()) < 1e-6
    moving = estep_residuals(mu, sigma, phi)
    assert moving["r_mu"].shape == (3, 6)                          # T = T+1 - 1


def test_spearman_rho_uses_fractional_average_ranks_for_ties():
    x = torch.tensor([1.0, 1.0, 2.0, 3.0])
    y = torch.tensor([1.0, 2.0, 2.0, 3.0])
    rank_x = torch.tensor([0.5, 0.5, 2.0, 3.0], dtype=torch.float64)
    rank_y = torch.tensor([0.0, 1.5, 1.5, 3.0], dtype=torch.float64)
    expected = float(torch.corrcoef(torch.stack([rank_x, rank_y]))[0, 1])

    assert abs(spearman_rho(x, y) - expected) < 1e-12


def test_spearman_rho_filters_paired_nonfinite_entries():
    x = torch.tensor([1.0, float("nan"), 2.0, 3.0, float("inf"), 4.0])
    y = torch.tensor([4.0, 99.0, 3.0, 2.0, 1.0, float("nan")])

    assert abs(spearman_rho(x, y) + 1.0) < 1e-12


def test_spearman_rho_returns_nan_with_fewer_than_two_finite_pairs():
    rho = spearman_rho(
        torch.tensor([1.0, float("nan"), float("inf")]),
        torch.tensor([2.0, 3.0, 4.0]),
    )

    assert math.isnan(rho)


def test_spearman_rho_returns_zero_for_constant_finite_input():
    assert spearman_rho(torch.ones(4), torch.arange(4.0)) == 0.0
    assert spearman_rho(torch.arange(4.0), torch.ones(4)) == 0.0


def test_spearman_rho_non_degenerate_ordering_is_unchanged():
    x = torch.arange(5.0)
    assert abs(spearman_rho(x, 2.0 * x + 1.0) - 1.0) < 1e-12
    assert abs(spearman_rho(x, -x) + 1.0) < 1e-12


def test_bootstrap_ce_band_brackets_point():
    nats = torch.rand(40) * 5.0 + 1.0
    toks = torch.full((40,), 10.0)
    band = bootstrap_ce_band(nats, toks, n_boot=500, seed=0)
    assert band["lo"] <= band["ce"] <= band["hi"]


def test_bootstrap_ce_band_returns_nan_for_zero_total_tokens():
    band = bootstrap_ce_band(
        torch.tensor([2.0, 3.0, 4.0]),
        torch.zeros(3),
        n_boot=20,
        seed=0,
    )

    assert all(math.isnan(band[key]) for key in ("ce", "lo", "hi"))


def test_bootstrap_token_ce_band_paired_zero_when_equal():
    a = torch.rand(200)
    band = bootstrap_token_ce_band(a, a, n_boot=300, seed=0)
    assert abs(band["delta"]) < 1e-6 and abs(band["lo"]) < 1e-6 and abs(band["hi"]) < 1e-6


def test_gauge_equivariance_residual_in_vs_out_group():
    torch.manual_seed(0)
    grp = get_group("block_glk")(4, 2)
    n_gen = grp.generators.shape[0]
    omega = compute_transport_operators(0.2 * torch.randn(1, 5, n_gen), grp)["Omega"][0]
    mu = torch.randn(5, 4)
    sigma = torch.rand(5, 4) + 0.5
    out = gauge_equivariance_residual(mu, sigma, omega, grp, n_samples=4, scale=0.4, seed=0)
    assert float(out["energy_in_group"].max()) < 1e-2             # invariant to machine tolerance
    assert float(out["energy_out_group"].mean()) > 10 * float(out["energy_in_group"].mean() + 1e-12)
