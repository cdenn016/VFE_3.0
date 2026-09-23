r"""Group C remediation of the 2026-09-23 deep audit: geometry, numerics, inference, families.

Every model and tensor here is TINY (single-digit dims) and CPU-bound. Host reads are counted by
wrapping the Python-level ``torch.Tensor`` conversions (``bool``/``float``/``int``/``item``/
``tolist``), which are exactly the calls that synchronize with a CUDA device.
"""

import inspect
import math
import warnings
from typing import Optional

import pytest
import torch

import vfe3.geometry.retraction as retraction
import vfe3.geometry.transport as transport
import vfe3.relational_scoring as relational_scoring
from tests.profiles import tiny_config
from vfe3.belief import BeliefState
from vfe3.families.exact_congruence import ExactCongruenceDiagonalGaussian
from vfe3.gauge_optim import GaugeManifoldAdamW, stage_pullback_group_candidate
from vfe3.geometry.groups import get_group
from vfe3.geometry.lie_ops import _from_equal_diag_blocks, embed_phi, extract_phi
from vfe3.geometry.phi_preconditioner import _bch_chart_direction, pullback_group_direction
from vfe3.geometry.retraction import retract_phi
from vfe3.geometry.rope import build_rope_rotation
from vfe3.geometry.transport import (
    CompactFactoredTransport,
    FactoredTransport,
    RopeTransport,
    _checked_group_inverse,
    stable_matrix_exp_pair,
)
from vfe3.inference import e_step as e_step_module
from vfe3.inference.e_step import e_step_iteration, free_energy_value, phi_alignment_loss
from vfe3.model.model import VFEModel
from vfe3.numerics import condition_number, floor_eigenvalues
from vfe3.relational_scoring import (
    projected_gaussian_kl,
    projected_moments,
    source_recovery_counter,
)


class _HostReads:
    r"""Count Python-level tensor-to-host conversions while installed."""

    NAMES = ("__bool__", "__float__", "__int__", "item", "tolist")

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.count = 0
        for name in self.NAMES:
            original = getattr(torch.Tensor, name)

            def wrapper(tensor, *args, _original=original, **kwargs):
                self.count += 1
                return _original(tensor, *args, **kwargs)

            monkeypatch.setattr(torch.Tensor, name, wrapper)


# -- #5: _checked_group_inverse pays ONE host read per strict call -------------------------------

def test_checked_group_inverse_strict_route_pays_one_host_read(monkeypatch) -> None:
    torch.manual_seed(0)
    omega = torch.linalg.matrix_exp(0.3 * torch.randn(3, 2, 4, 4))
    reference = torch.linalg.inv(omega.double()).to(omega.dtype)

    def _no_linalg_inv(*args, **kwargs):
        raise AssertionError("torch.linalg.inv checks its LU status with its own host read")

    monkeypatch.setattr(torch.linalg, "inv", _no_linalg_inv)
    reads = _HostReads(monkeypatch)
    got = _checked_group_inverse(omega)
    assert reads.count == 1
    assert torch.equal(got, reference)

    reads.count = 0
    _checked_group_inverse(omega, residual_tol=1e-4)
    assert reads.count == 1


def test_checked_group_inverse_still_fails_closed_in_order() -> None:
    bad = torch.eye(3).expand(2, 3, 3).clone()
    bad[1, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="nonfinite values before inversion"):
        _checked_group_inverse(bad)
    singular = torch.zeros(2, 3, 3, dtype=torch.float64)
    with pytest.raises(ValueError, match="singular"):
        _checked_group_inverse(singular)
    drifted = torch.eye(3, dtype=torch.float32) * 1e-3
    drifted[0, 1] = 1e3                   # cond ~ 1e12: the stored float32 pair is not inverse
    with pytest.raises(FloatingPointError, match="violates the inverse identity"):
        _checked_group_inverse(drifted, residual_tol=1e-9)


# -- Low: the clamp-surrogate latch no longer taxes every well-conditioned build ----------------

def test_stable_matrix_exp_pair_reads_host_once_per_decision_and_still_warns_first_clamp(
    monkeypatch,
) -> None:
    monkeypatch.setattr(transport, "_CLAMP_SURROGATE_WARNED", False)
    torch.manual_seed(0)
    small = 0.3 * torch.randn(4, 3, 3)
    reads = _HostReads(monkeypatch)
    counts = []
    for _ in range(3):
        reads.count = 0
        stable_matrix_exp_pair(small)
        counts.append(reads.count)
    # One read for the pre-exponential decisions, one for the strict inverse; never a third.
    assert counts == [2, 2, 2]
    assert transport._CLAMP_SURROGATE_WARNED is False

    big = 40.0 * torch.eye(3).expand(2, 3, 3)
    with pytest.warns(RuntimeWarning, match="SURROGATE"):
        stable_matrix_exp_pair(big)
    assert transport._CLAMP_SURROGATE_WARNED is True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reads.count = 0
        stable_matrix_exp_pair(big)
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert reads.count == 2


def test_stable_matrix_exp_pair_validity_bound_still_raises() -> None:
    with pytest.raises(ValueError, match="validity bound exceeded"):
        stable_matrix_exp_pair(10.0 * torch.eye(3).unsqueeze(0), validity_max_norm=1.0)


# -- #6: the filtering phi objective is causal ---------------------------------------------------

def _max_future_leak(cfg) -> float:
    torch.manual_seed(0)
    model = VFEModel(cfg).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.dtype.is_floating_point and any(
                    key in name for key in ("phi_embed", "mu_embed", "pos_phi")):
                parameter.add_(0.3 * torch.randn_like(parameter))
    generator = torch.Generator().manual_seed(1)
    V, N = cfg.vocab_size, cfg.max_seq_len
    tokens = torch.randint(0, V, (2, N), generator=generator)
    worst = 0.0
    with torch.no_grad():
        reference = model(tokens)
        for t in range(N - 1):
            resampled = tokens.clone()
            resampled[:, t + 1:] = torch.randint(0, V, (2, N - 1 - t), generator=generator)
            logits = model(resampled)
            worst = max(worst, (reference[:, :t + 1] - logits[:, :t + 1]).abs().max().item())
    return worst


@pytest.mark.parametrize("depth", [dict(n_layers=2), dict(n_layers=1, n_e_steps=2)],
                         ids=["two_layers", "one_layer_two_steps"])
def test_transported_phi_estep_does_not_leak_future_tokens(depth: dict) -> None:
    cfg = tiny_config(attention_mode="transported", beta_attention_prior="causal",
                      vocab_size=13, max_seq_len=6, e_phi_lr=0.05, **depth)
    assert cfg.e_phi_lr > 0.0 and cfg.gradient_mode == "filtering"
    # Before the fix this measured 3.0e-3 on this exact configuration.
    assert _max_future_leak(cfg) <= 1e-6


def _phi_objective_inputs(seed: int = 0):
    group = get_group("block_glk")(4, 2)
    generator = torch.Generator().manual_seed(seed)
    B, N, K = 1, 5, 4
    mu = torch.randn(B, N, K, generator=generator)
    sigma = torch.rand(B, N, K, generator=generator) + 0.5
    phi = 0.4 * torch.randn(B, N, group.generators.shape[0], generator=generator)
    log_prior = torch.zeros(N, N).masked_fill(torch.ones(N, N).triu(1).bool(), float("-inf"))
    return group, mu, sigma, phi, log_prior


@pytest.mark.parametrize("family", ["gaussian_diagonal", "gaussian_diagonal_exact"])
@pytest.mark.parametrize("compact", [False, True])
def test_filtering_phi_gradient_uses_only_receiver_rows(family: str, compact: bool) -> None:
    group, mu, sigma, phi, log_prior = _phi_objective_inputs()

    def gradient(detach: bool, mu_, sigma_, phi_) -> torch.Tensor:
        leaf = phi_.clone().requires_grad_(True)
        loss = phi_alignment_loss(mu_, sigma_, leaf, group, tau=1.0, log_prior=log_prior,
                                  family=family, compact_phi_block_transport=compact,
                                  detach_key_frames=detach)
        return torch.autograd.grad(loss, leaf)[0]

    mu2, sigma2, phi2 = mu.clone(), sigma.clone(), phi.clone()
    mu2[:, -1] += 1.0
    sigma2[:, -1] *= 2.0
    phi2[:, -1] += 0.3
    receiver = gradient(True, mu, sigma, phi)
    receiver_future = gradient(True, mu2, sigma2, phi2)
    assert torch.isfinite(receiver).all()
    assert torch.equal(receiver[:, :-1], receiver_future[:, :-1])
    # Without the detach, earlier frames see the later receiver row: the leak the fix removes.
    full = gradient(False, mu, sigma, phi)
    full_future = gradient(False, mu2, sigma2, phi2)
    assert (full[:, :-1] - full_future[:, :-1]).abs().max() > 1e-4
    # The objective VALUE is unchanged by the detach.
    kwargs = dict(tau=1.0, log_prior=log_prior, family=family, compact_phi_block_transport=compact)
    assert torch.allclose(phi_alignment_loss(mu, sigma, phi, group, detach_key_frames=True, **kwargs),
                          phi_alignment_loss(mu, sigma, phi, group, **kwargs), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("gradient_mode, expected", [("filtering", True), ("smoothing", False)])
def test_phi_objective_detach_follows_gradient_mode(monkeypatch, gradient_mode: str,
                                                    expected: bool) -> None:
    group, mu, sigma, phi, log_prior = _phi_objective_inputs()
    seen = []
    original = e_step_module.phi_alignment_loss

    def spy(*args, **kwargs):
        seen.append(kwargs.get("detach_key_frames", False))
        return original(*args, **kwargs)

    monkeypatch.setattr(e_step_module, "phi_alignment_loss", spy)
    belief = BeliefState(mu=mu[0], sigma=sigma[0], phi=phi[0])
    e_step_iteration(belief, mu[0].clone(), sigma[0].clone(), group, e_phi_lr=0.01,
                     e_q_mu_lr=0.0, e_q_sigma_lr=0.0, gradient_mode=gradient_mode,
                     log_prior=log_prior, sigma_max=10.0)
    assert seen == [expected]


# -- #7: the E-step phi retraction bounds the step, not the base point ---------------------------

@pytest.mark.parametrize("group_name, mode, compact", [
    ("glk", "euclidean", False), ("glk", "bch", False), ("block_glk", "bch", True)])
def test_retract_phi_zero_step_returns_an_exterior_frame(group_name: str, mode: str,
                                                          compact: bool) -> None:
    group = (get_group(group_name)(4, 2, dtype=torch.float64) if group_name == "block_glk"
             else get_group(group_name)(3, dtype=torch.float64))
    torch.manual_seed(0)
    phi = torch.randn(3, group.generators.shape[0], dtype=torch.float64)
    phi = 8.0 * phi / phi.norm(dim=-1, keepdim=True)              # outside the default cap of 5
    out = retract_phi(phi, torch.zeros_like(phi), group, mode=mode, compact_blocks=compact)
    if mode == "euclidean" or compact:
        assert torch.equal(out, phi)
    else:
        assert torch.allclose(out, phi, rtol=0.0, atol=1e-12)


def test_retract_phi_keeps_identity_differential_inside_and_bounds_outward_steps() -> None:
    group = get_group("glk")(3, dtype=torch.float64)
    torch.manual_seed(1)
    inside = torch.randn(2, 9, dtype=torch.float64)
    inside = 1.5 * inside / inside.norm(dim=-1, keepdim=True)
    direction = torch.randn(2, 9, dtype=torch.float64)
    t = 1e-7
    step = retract_phi(inside, direction, group, step_size=t, trust_region=100.0)
    assert torch.allclose((step - inside) / t, direction, atol=1e-6)

    outside = 8.0 * inside / inside.norm(dim=-1, keepdim=True)
    moved = retract_phi(outside, outside, group, step_size=1.0, trust_region=100.0)
    assert (moved.norm(dim=-1) <= 8.0 + 1e-9).all()
    # A step from inside the ball still saturates at the group constant.
    saturated = retract_phi(torch.zeros(2, 9, dtype=torch.float64),
                            1e3 * torch.ones(2, 9, dtype=torch.float64), group, trust_region=100.0)
    assert torch.allclose(saturated.norm(dim=-1), torch.full((2,), 5.0, dtype=torch.float64),
                          atol=1e-3)


def test_estep_phi_step_no_longer_collapses_large_frames() -> None:
    group = get_group("block_glk")(4, 2)
    generator = torch.Generator().manual_seed(0)
    N = 3
    phi = torch.randn(N, group.generators.shape[0], generator=generator)
    phi = 8.0 * phi / phi.norm(dim=-1, keepdim=True)
    belief = BeliefState(mu=torch.randn(N, 4, generator=generator),
                         sigma=torch.rand(N, 4, generator=generator) + 0.5, phi=phi)
    out = e_step_iteration(belief, belief.mu.clone(), belief.sigma.clone(), group,
                           e_phi_lr=1e-4, e_q_mu_lr=0.0, e_q_sigma_lr=0.0,
                           phi_retract_mode="bch", sigma_max=10.0)
    displacement = (out.phi - phi).norm(dim=-1) / phi.norm(dim=-1)
    assert displacement.max() < 1e-2                  # was 0.375 (norm 8 -> 5) before the fix


# -- #10: pullback_group rejects ill-conditioned or chart-violating rows individually ------------

def test_pullback_staging_rejects_only_the_ill_conditioned_row(monkeypatch) -> None:
    group = get_group("glk")(K=2, dtype=torch.float64)
    bad = torch.tensor([5.0 / math.sqrt(2.0), 0.0, 0.0, -5.0 / math.sqrt(2.0)], dtype=torch.float64)
    good = torch.tensor([0.1, 0.05, -0.02, 0.03], dtype=torch.float64)
    phi = torch.stack((bad, good))
    grad = torch.ones_like(phi)
    with pytest.raises(ValueError, match="damped generalized condition"):
        pullback_group_direction(grad, phi, group.generators, mode="pullback")

    candidate = stage_pullback_group_candidate(
        grad, phi, group, learning_rate=1e-3, trust_radius=0.1, chart_max_norm=10.0,
        bch_residual_max=1e-6, phi_precond_mode="pullback")
    assert candidate.rejected_rows.tolist() == [True, False]
    assert torch.equal(candidate.candidate_phi[0], phi[0])
    assert not torch.equal(candidate.candidate_phi[1], phi[1])

    param = torch.nn.Parameter(phi.clone())
    optimizer = GaugeManifoldAdamW(
        [{"params": [param], "lr": 1e-3, "pullback_group": True, "weight_decay": 0.0}], group,
        phi_group_trust_radius=0.1, phi_chart_max_norm=10.0, phi_bch_residual_max=1e-6,
        phi_precond_mode="pullback", weight_decay=0.0)
    param.grad = grad.clone()
    optimizer._collect_gauge_diag = True
    optimizer.step()
    assert torch.equal(param.detach()[0], phi[0])
    assert torch.equal(param.detach()[1], candidate.candidate_phi[1])
    assert optimizer.phi_rejected_rows == 1
    assert optimizer._gauge_diag["phi_group_rejected_rows"] == 1.0


def test_pullback_staging_backtracks_or_rejects_chart_bound_rows_per_row(monkeypatch) -> None:
    import vfe3.gauge_optim as gauge_optim
    from vfe3.geometry.phi_preconditioner import PullbackGroupDirectionResult

    group = get_group("glk")(K=2, dtype=torch.float64)
    phi = torch.tensor([[0.99, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    xi = torch.tensor([[-0.2, 0.0, 0.0, 0.0]] * 2, dtype=torch.float64)
    ones, zeros = torch.ones(2, dtype=torch.float64), torch.zeros(2, dtype=torch.float64)
    monkeypatch.setattr(gauge_optim, "pullback_group_direction", lambda *a, **k: (
        PullbackGroupDirectionResult(v_phi=xi, xi=xi, min_undamped_generalized_eigenvalue=ones,
                                     undamped_generalized_condition=ones,
                                     damped_generalized_condition=ones,
                                     scaled_solve_residual=zeros, series_order=40)))
    candidate = stage_pullback_group_candidate(
        -torch.ones_like(phi), phi, group, learning_rate=1.0, trust_radius=1.0,
        chart_max_norm=1.0, bch_residual_max=1e-6, phi_precond_mode="pullback")
    # Row 0 starts inside: 0.99 + 0.2 / 2^5 = 0.99625 is the first backtrack inside the chart.
    assert candidate.backtracking_reductions[0].item() == 5
    assert candidate.candidate_phi[0, 0].item() == pytest.approx(0.99625)
    # Row 1 sits ON the bound and the step is outward: no backtrack returns inside, so it alone
    # keeps its chart instead of aborting the step.
    assert candidate.rejected_rows.tolist() == [False, True]
    assert torch.equal(candidate.candidate_phi[1], phi[1])
    assert (candidate.candidate_chart_norm <= 1.0).all()


def test_pullback_staging_still_raises_on_nonfinite_state() -> None:
    group = get_group("glk")(K=2, dtype=torch.float64)
    phi = torch.zeros(2, 4, dtype=torch.float64)
    grad = torch.ones_like(phi)
    grad[0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="nonfinite grad or chart"):
        stage_pullback_group_candidate(
            grad, phi, group, learning_rate=1e-3, trust_radius=0.1, chart_max_norm=10.0,
            bch_residual_max=1e-6, phi_precond_mode="pullback")


# -- #11: the E-step chart direction uses the exact exponential differential ---------------------

def _psi(z: torch.Tensor) -> torch.Tensor:
    r"""psi(z) = z / (1 - e^{-z}), continued to psi(0) = 1."""
    safe = torch.where(z.abs() < 1e-12, torch.ones_like(z), z)
    return torch.where(z.abs() < 1e-12, torch.ones_like(z), safe / (-torch.expm1(-safe)))


def test_bch_chart_direction_is_exact_at_the_spurious_b4_root() -> None:
    group = get_group("glk")(2)
    G = group.generators.double()
    a = 3.879 / 2.0                                               # ad eigenvalues 0, +-3.879
    phi = extract_phi(torch.diag(torch.tensor([a, -a], dtype=torch.float64)), G)
    delta = torch.tensor([0.3, -0.7, 0.4, 0.2], dtype=torch.float64)
    v = _bch_chart_direction(delta, phi, G)
    assert torch.isfinite(v).all()
    lam = torch.tensor([a, -a], dtype=torch.float64)
    z = lam.unsqueeze(-1) - lam.unsqueeze(-2)                     # ad eigenvalue on E_ij
    V, D = embed_phi(v, G), embed_phi(delta, G)
    # Exact: psi(ad_X) v = delta entrywise in the eigenbasis of the diagonal X.
    assert torch.allclose(_psi(z) * V, D, rtol=0.0, atol=1e-12)
    # The retired B4 truncation is (nearly) singular here; the exact direction stays O(delta).
    assert v.norm() < 20.0 * delta.norm()


def test_bch_chart_direction_matches_the_exact_group_product_velocity() -> None:
    scipy_linalg = pytest.importorskip("scipy.linalg")
    group = get_group("glk")(3)
    G = group.generators.double()
    generator = torch.Generator().manual_seed(3)
    phi = 0.8 * torch.randn(9, generator=generator, dtype=torch.float64)
    delta = torch.randn(9, generator=generator, dtype=torch.float64)
    v = _bch_chart_direction(delta, phi, G)
    X = embed_phi(phi, G).numpy()
    V = embed_phi(v, G).numpy()
    t = 1e-5
    forward = scipy_linalg.logm(scipy_linalg.expm(X) @ scipy_linalg.expm(t * V))
    backward = scipy_linalg.logm(scipy_linalg.expm(X) @ scipy_linalg.expm(-t * V))
    velocity = torch.from_numpy(((forward - backward) / (2 * t)).real)
    assert torch.allclose(velocity, embed_phi(delta, G), atol=1e-6)


def test_bch_chart_direction_per_block_matches_full_on_block_glk() -> None:
    group = get_group("block_glk")(4, 2)
    G = group.generators.double()
    generator = torch.Generator().manual_seed(4)
    phi = torch.randn(3, G.shape[0], generator=generator, dtype=torch.float64)
    delta = torch.randn(3, G.shape[0], generator=generator, dtype=torch.float64)
    per_block = _bch_chart_direction(delta, phi, G, irrep_dims=group.irrep_dims)
    full = _bch_chart_direction(delta, phi, G)
    assert torch.allclose(per_block, full, atol=1e-10)


def test_estep_bch_residual_gate_reaches_compose_bch() -> None:
    group = get_group("glk")(2)
    generator = torch.Generator().manual_seed(5)
    N = 3
    belief = BeliefState(mu=torch.randn(N, 2, generator=generator),
                         sigma=torch.rand(N, 2, generator=generator) + 0.5,
                         phi=1.5 * torch.randn(N, 4, generator=generator))
    kwargs = dict(e_phi_lr=5.0, e_q_mu_lr=0.0, e_q_sigma_lr=0.0, phi_retract_mode="bch",
                  sigma_max=10.0)
    e_step_iteration(belief, belief.mu.clone(), belief.sigma.clone(), group, **kwargs)
    with pytest.raises(ValueError, match="BCH residual validity bound exceeded"):
        e_step_iteration(belief, belief.mu.clone(), belief.sigma.clone(), group,
                         bch_residual_max=1e-12, **kwargs)
    # The shared knob bag also reaches the diagnostic F, which accepts and ignores it.
    assert torch.isfinite(free_energy_value(belief, belief.mu, belief.sigma, group,
                                            bch_residual_max=1e-12))


# -- #21: gaussian_diagonal_exact under means-only RoPE is the coherent exact KL -----------------

def _dense_kl(mu_q, sigma_q, mu_p, sigma_p) -> torch.Tensor:
    precision = torch.linalg.inv(sigma_p)
    delta = (mu_p - mu_q).unsqueeze(-1)
    return 0.5 * (torch.trace(precision @ sigma_q) + (delta.T @ precision @ delta).squeeze()
                  - mu_q.shape[-1] + torch.logdet(sigma_p) - torch.logdet(sigma_q))


@pytest.mark.parametrize("representation", ["factored", "compact"])
def test_exact_family_under_rope_equals_the_dense_rotated_pushforward_kl(
        representation: str) -> None:
    torch.manual_seed(0)
    n, n_blocks, d = 4, 2, 2
    K = n_blocks * d
    blocks = torch.linalg.matrix_exp(0.5 * torch.randn(n, n_blocks, d, d, dtype=torch.float64))
    inv_blocks = torch.linalg.inv(blocks)
    U, U_inv = _from_equal_diag_blocks(blocks, K), _from_equal_diag_blocks(inv_blocks, K)
    base = (FactoredTransport(exp_phi=U, exp_neg_phi=U_inv, irrep_dims=[d] * n_blocks,
                              same_frame_flat_cocycle=True)
            if representation == "factored"
            else CompactFactoredTransport(blocks, inv_blocks, K, same_frame_flat_cocycle=True))
    rope = build_rope_rotation(torch.arange(n), [d] * n_blocks, base=3.0, dtype=torch.float64)
    omega = RopeTransport(base=base, rope=rope, on_cov=False, insertion="right",
                          same_frame_flat_cocycle=True)
    mu = torch.randn(n, K, dtype=torch.float64)
    sigma = torch.rand(n, K, dtype=torch.float64) + 0.4
    energy = ExactCongruenceDiagonalGaussian.coupling_energy(
        mu, sigma, mu, sigma, omega, kl_max=1e9, eps=1e-12, irrep_dims=[d] * n_blocks)
    V = U @ rope.transpose(-1, -2)                                # folded frame V_i = U_i R_i^T
    dense = torch.einsum("ikl,jlm->ijkm", V, torch.linalg.inv(V))
    worst = 0.0
    for h in range(n_blocks):
        block = slice(h * d, (h + 1) * d)
        for i in range(n):
            for j in range(n):
                link = dense[i, j][block, block]
                reference = _dense_kl(mu[i][block], torch.diag(sigma[i][block]),
                                      link @ mu[j][block],
                                      link @ torch.diag(sigma[j][block]) @ link.T)
                worst = max(worst, abs(energy[h, i, j].item() - reference.item()))
    assert worst <= 1e-10


def test_exact_family_left_rope_fails_closed_and_split_frames_use_the_dense_route() -> None:
    torch.manual_seed(1)
    n, K = 3, 4
    U = torch.linalg.matrix_exp(0.3 * torch.randn(n, K, K, dtype=torch.float64))
    base = FactoredTransport(exp_phi=U, exp_neg_phi=torch.linalg.inv(U), irrep_dims=[K],
                             same_frame_flat_cocycle=True)
    rope = build_rope_rotation(torch.arange(n), [K], dtype=torch.float64)
    mu = torch.randn(n, K, dtype=torch.float64)
    sigma = torch.rand(n, K, dtype=torch.float64) + 0.4
    left = RopeTransport(base=base, rope=rope, insertion="left", same_frame_flat_cocycle=True)
    with pytest.raises(TypeError, match="left-insertion"):
        ExactCongruenceDiagonalGaussian.coupling_energy(mu, sigma, mu, sigma, left)
    split = FactoredTransport(exp_phi=U, exp_neg_phi=torch.linalg.inv(U).flip(0), irrep_dims=[K],
                              same_frame_flat_cocycle=False)
    energy = ExactCongruenceDiagonalGaussian.coupling_energy(mu, sigma, mu, sigma, split)
    dense = ExactCongruenceDiagonalGaussian.coupling_energy(mu, sigma, mu, sigma,
                                                            split.to_dense_omega())
    assert torch.equal(energy, dense)


# -- #27: sigma_max annotations match VFE3Config ------------------------------------------------

@pytest.mark.parametrize("function", [e_step_iteration, free_energy_value])
def test_sigma_max_annotation_matches_config(function) -> None:
    parameter = inspect.signature(function).parameters["sigma_max"]
    assert parameter.annotation == Optional[float]
    assert parameter.default == tiny_config().sigma_max == 100.0


# -- Low: preconditioner labels ----------------------------------------------------------------

def test_pullback_and_killing_preconditioners_are_not_labeled_natural_gradient() -> None:
    import vfe3.geometry.phi_preconditioner as pp

    for function in (pp.pullback_metric, pp._precond_pullback, pp._precond_pullback_per_block,
                     pp.pullback_metric_per_block, pp._precond_killing,
                     pp._precond_killing_per_block):
        assert "natural" not in (function.__doc__ or "").lower()


# -- Low: numerics keep float64 in float64 -------------------------------------------------------

def test_condition_number_and_floor_eigenvalues_keep_float64() -> None:
    torch.manual_seed(0)
    Q = torch.linalg.qr(torch.randn(8, 8, dtype=torch.float64))[0]
    spectrum = torch.logspace(-10, 0, 8, dtype=torch.float64)
    A = Q @ torch.diag(spectrum) @ Q.T
    cond = condition_number(A)
    assert cond.dtype == torch.float64
    assert cond.item() == pytest.approx(1e10, rel=1e-3)          # float32 returned inf here
    floored = floor_eigenvalues(A, floor=1e-9)
    assert floored.dtype == torch.float64
    assert torch.linalg.eigvalsh(floored).min().item() >= 1e-9 * (1 - 1e-6)
    assert condition_number(A.float()).dtype == torch.float32    # float32 policy unchanged


def test_eigh_damped_forward_routes_through_safe_eigh(monkeypatch) -> None:
    calls = []
    original = retraction.safe_eigh

    def spy(matrix, **kwargs):
        calls.append(matrix.shape)
        return original(matrix, **kwargs)

    monkeypatch.setattr(retraction, "safe_eigh", spy)
    A = torch.randn(3, 4, 4, dtype=torch.float64)
    A = A @ A.transpose(-1, -2) + torch.eye(4, dtype=torch.float64)
    w, V = retraction._eigh_damped(A, 1e-12)
    reference = torch.linalg.eigh(A)
    assert calls == [A.shape]
    assert torch.equal(w, reference.eigenvalues) and torch.equal(V, reference.eigenvectors)


# -- Low: relational scorer recovery counter, per-pair routing, exact certificate ----------------

def _projected_inputs(ill: bool, batch: int = 2, tokens: int = 6, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    H, d = 1, 3
    mu = torch.randn(batch, tokens, H, d, generator=generator)
    sigma = torch.rand(batch, tokens, H, d, generator=generator) + 0.5
    if ill:
        sigma[1, tokens - 1, 0] = torch.tensor([1.0, 1.0, 3e-5])  # certificate-true, recovered
    inverse = torch.eye(d).expand(batch, tokens, H, d, d).contiguous()
    S_q = torch.linalg.qr(torch.randn(d, d, generator=generator))[0].T.unsqueeze(0).contiguous()
    S_k = torch.linalg.qr(torch.randn(d, d, generator=generator))[0].T.unsqueeze(0).contiguous()
    return mu, sigma, inverse, S_q, S_k


def _score(mu, sigma, inverse, S_q, S_k) -> torch.Tensor:
    a_q, G_q = projected_moments(mu, sigma, inverse, S_q)
    a_k, G_k = projected_moments(mu, sigma, inverse, S_k)
    return projected_gaussian_kl(a_q, G_q, a_k, G_k,
                                 projection_q=(sigma, inverse, S_q, mu),
                                 projection_k=(sigma, inverse, S_k, mu))


@pytest.mark.usefixtures("condition_recovery_on")
def test_factor_route_is_selected_per_pair_and_counted() -> None:
    healthy_inputs = _projected_inputs(ill=False)
    ill_inputs = _projected_inputs(ill=True)
    device = healthy_inputs[0].device
    before = source_recovery_counter(device)
    healthy = _score(*healthy_inputs)
    assert torch.equal(source_recovery_counter(device), before)
    ill = _score(*ill_inputs)
    recovered = int(source_recovery_counter(device) - before)
    assert recovered >= 1                                          # the ill token, both roles
    assert torch.isfinite(ill).all() and ill.dtype == torch.float32
    # Another sequence's recovered token leaves sequence 0 bit-identical.
    assert torch.equal(ill[0], healthy[0])
    # A later recovered token leaves every earlier pair of its own sequence bit-identical.
    assert torch.equal(ill[1, :, :-1, :-1], healthy[1, :, :-1, :-1])


def test_projected_kl_healthy_path_reads_host_once_even_when_moderately_conditioned(
        monkeypatch) -> None:
    generator = torch.Generator().manual_seed(2)
    B, N, H, d = 1, 5, 1, 4
    mu = torch.randn(B, N, H, d, generator=generator)
    # kappa ~ 300: the retired AM-GM screen fired here, the exact certificate does not.
    sigma = torch.tensor([1.0, 0.3, 0.05, 1.0 / 300.0]).expand(B, N, H, d).contiguous()
    inverse = torch.eye(d).expand(B, N, H, d, d).contiguous()
    S = torch.eye(d).unsqueeze(0)
    a, G = projected_moments(mu, sigma, inverse, S)
    status = relational_scoring._cholesky_status(G)
    assert not bool(status[2].any())                               # no entry needs recovery
    reads = _HostReads(monkeypatch)
    projected_gaussian_kl(a, G, a, G, projection_q=(sigma, inverse, S, mu),
                          projection_k=(sigma, inverse, S, mu))
    assert reads.count == 1


def test_estep_capture_publishes_the_recovery_count(monkeypatch) -> None:
    import itertools

    from vfe3.config import VFE3Config

    cfg = VFE3Config(vocab_size=7, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=1,
                     n_e_steps=1, family="gaussian_diagonal", gauge_group="block_glk",
                     skip_belief_sigma_update=True, e_phi_lr=0.0, oracle_unroll_grad=True,
                     warmup_steps=0, max_steps=2, log_interval=0, eval_interval=0,
                     pos_phi="learned", use_head_mixer=True, use_block_mlp=True,
                     block_mlp_expansion=1, attention_mode="projected_conditional")
    torch.manual_seed(0)
    model = VFEModel(cfg)
    tokens = torch.tensor([[0, 1, 2, 3]])
    diagnostics = model.diagnostics(tokens)
    assert diagnostics["relational_covariance_repair_count"] == 0.0

    # Each E-step iteration publishes (after - before) of the on-device counter.
    counts = itertools.cycle((torch.tensor(5), torch.tensor(7)))
    monkeypatch.setattr(e_step_module, "source_recovery_counter", lambda device: next(counts))
    diagnostics = model.diagnostics(tokens)
    assert diagnostics["relational_covariance_repair_count"] == 2.0
