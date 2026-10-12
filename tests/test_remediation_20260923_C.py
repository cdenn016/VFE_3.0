r"""Group C remediation of the 2026-09-23 deep audit: geometry, numerics, inference, families.

Every model and tensor here is TINY (single-digit dims) and CPU-bound. Host reads are counted by
wrapping the Python-level ``torch.Tensor`` conversions (``bool``/``float``/``int``/``item``/
``tolist``), which are exactly the calls that synchronize with a CUDA device.
"""
from tests.profiles import test_config

import inspect
import math
from typing import Optional

import pytest
import torch

import vfe3.geometry.retraction as retraction
import vfe3.geometry.transport as transport
import vfe3.relational_scoring as relational_scoring
from tests.profiles import tiny_config
from vfe3.belief import BeliefState
from vfe3.gauge_optim import GaugeManifoldAdamW, stage_pullback_group_candidate
from vfe3.geometry.groups import get_group
from vfe3.geometry.phi_preconditioner import pullback_group_direction
from vfe3.geometry.transport import (
    _checked_group_inverse,
    stable_matrix_exp_pair,
)
from vfe3.inference import e_step as e_step_module
from vfe3.inference.e_step import e_step_iteration
from vfe3.model.model import VFEModel
from vfe3.relational_scoring import (
    projected_gaussian_kl,
    projected_moments,
    source_recovery_counter,
)


def _pullback_single_block(grad_phi, phi, generators, **kwargs):
    """Strict pullback direction on one block (the retired ambient 'pullback' mode's live equivalent)."""
    kwargs.setdefault("irrep_dims", [generators.shape[-1]])
    return pullback_group_direction(grad_phi, phi, generators, mode="pullback_per_block", **kwargs)


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


# -- Low: one host read for the pre-exponential chart decisions, one for the strict inverse -------

def test_stable_matrix_exp_pair_reads_host_once_per_decision(monkeypatch) -> None:
    torch.manual_seed(0)
    small = 0.3 * torch.randn(4, 3, 3)
    reads = _HostReads(monkeypatch)
    counts = []
    for _ in range(3):
        reads.count = 0
        stable_matrix_exp_pair(small)
        counts.append(reads.count)
    assert counts == [2, 2, 2]


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


# -- #10: pullback_group rejects ill-conditioned or chart-violating rows individually ------------

def test_pullback_staging_rejects_only_the_ill_conditioned_row(monkeypatch) -> None:
    group = get_group("block_glk")(n_heads=1, K=2, dtype=torch.float64)
    bad = torch.tensor([5.0 / math.sqrt(2.0), 0.0, 0.0, -5.0 / math.sqrt(2.0)], dtype=torch.float64)
    good = torch.tensor([0.1, 0.05, -0.02, 0.03], dtype=torch.float64)
    phi = torch.stack((bad, good))
    grad = torch.ones_like(phi)
    with pytest.raises(ValueError, match="damped generalized condition"):
        _pullback_single_block(grad, phi, group.generators)

    candidate = stage_pullback_group_candidate(
        grad, phi, group, learning_rate=1e-3, trust_radius=0.1, chart_max_norm=10.0)
    assert candidate.rejected_rows.tolist() == [True, False]
    assert torch.equal(candidate.candidate_phi[0], phi[0])
    assert not torch.equal(candidate.candidate_phi[1], phi[1])

    param = torch.nn.Parameter(phi.clone())
    optimizer = GaugeManifoldAdamW(
        [{"params": [param], "lr": 1e-3, "pullback_group": True, "weight_decay": 0.0}], group,
        phi_group_trust_radius=0.1, phi_chart_max_norm=10.0, weight_decay=0.0)
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

    group = get_group("block_glk")(n_heads=1, K=2, dtype=torch.float64)
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
        chart_max_norm=1.0)
    # Row 0 starts inside: 0.99 + 0.2 / 2^5 = 0.99625 is the first backtrack inside the chart.
    assert candidate.backtracking_reductions[0].item() == 5
    assert candidate.candidate_phi[0, 0].item() == pytest.approx(0.99625)
    # Row 1 sits ON the bound and the step is outward: no backtrack returns inside, so it alone
    # keeps its chart instead of aborting the step.
    assert candidate.rejected_rows.tolist() == [False, True]
    assert torch.equal(candidate.candidate_phi[1], phi[1])
    assert (candidate.candidate_chart_norm <= 1.0).all()


def test_pullback_staging_still_raises_on_nonfinite_state() -> None:
    group = get_group("block_glk")(n_heads=1, K=2, dtype=torch.float64)
    phi = torch.zeros(2, 4, dtype=torch.float64)
    grad = torch.ones_like(phi)
    grad[0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="nonfinite grad or chart"):
        stage_pullback_group_candidate(
            grad, phi, group, learning_rate=1e-3, trust_radius=0.1, chart_max_norm=10.0)


# -- #27: sigma_max annotations match VFE3Config ------------------------------------------------

@pytest.mark.parametrize("function", [e_step_iteration])
def test_sigma_max_annotation_matches_config(function) -> None:
    parameter = inspect.signature(function).parameters["sigma_max"]
    assert parameter.annotation == Optional[float]
    assert parameter.default == tiny_config().sigma_max == 100.0


# -- Low: numerics keep float64 in float64 -------------------------------------------------------


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


    cfg = test_config(vocab_size=7, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=1,
                     n_e_steps=1, family="gaussian_diagonal", gauge_group="block_glk",
                     skip_belief_sigma_update=True,
                     warmup_steps=0, max_steps=2, log_interval=0, eval_interval=0,
                     pos_phi="learned", use_block_mlp=True,
                     block_mlp_expansion=1, attention_mode="projected_gradient")
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
