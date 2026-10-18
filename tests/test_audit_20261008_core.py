"""Remediation of the 2026-10-08 audit model and numerics core items (docs/audits/audit-2026-10-08.md).

Tiny CPU fixtures only (V=11, K=4, H=2, N=6 models; d <= 4 direct calls).
"""
from tests.profiles import make_relational_operators
from typing import Any, Iterator, List

import pytest
import torch

import vfe3.relational_scoring as scoring
from tests.test_audit_20261007_opus_fixes import B, N, V, _cfg, _minimal_agent, _rope_cfg
from vfe3.contracts import CanonicalFrameContext
from vfe3.model.model import VFEModel


@pytest.fixture
def _precision_high() -> Iterator[None]:
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    yield
    torch.set_float32_matmul_precision(previous)


def _einsum_precisions(monkeypatch: pytest.MonkeyPatch) -> List[str]:
    seen: List[str] = []
    real = torch.einsum

    def recording(*args: Any, **kwargs: Any) -> torch.Tensor:
        seen.append(torch.get_float32_matmul_precision())
        return real(*args, **kwargs)

    monkeypatch.setattr(torch, "einsum", recording)
    return seen


# -- P2-20: centered moment-matched spread ---------------------------------------------------------

def _spread_inputs(dtype: torch.dtype, offset: float) -> tuple:
    g = torch.Generator().manual_seed(3)
    H, n, d = 2, 8, 3
    logits = torch.randn(1, H, n, n, generator=g, dtype=torch.float64)
    logits = logits + 12.0 * torch.eye(n, dtype=torch.float64)                 # sharp, self-dominated rows
    logits = logits.masked_fill(torch.triu(torch.ones(n, n, dtype=torch.bool), 1), float("-inf"))
    b = torch.softmax(logits, dim=-1)                                          # (1, H, n, n)
    values = torch.randn(1, n, H, d, generator=g, dtype=torch.float64)
    values[:, 1:] += offset                                                    # far from token 0
    return b.to(dtype), values.to(dtype)


def _one_pass(b: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    second = torch.einsum("...hij,...jhst->...ihst", b, values.unsqueeze(-1) * values.unsqueeze(-2))
    mean = torch.einsum("...hij,...jhs->...ihs", b, values)
    return second - mean.unsqueeze(-1) * mean.unsqueeze(-2)


def test_centered_spread_equals_the_one_pass_moment_in_float64_and_is_accurate_in_float32(
        monkeypatch: pytest.MonkeyPatch) -> None:
    from vfe3.model import relational_operators as ro

    b64, v64 = _spread_inputs(torch.float64, 0.0)
    mean64 = torch.einsum("...hij,...jhs->...ihs", b64, v64)
    torch.testing.assert_close(ro._centered_spread(b64, v64, mean64), _one_pass(b64, v64), rtol=1e-12, atol=1e-12)
    monkeypatch.setattr(ro, "_SPREAD_TILE_BYTES", 1)                           # one row per tile
    torch.testing.assert_close(ro._centered_spread(b64, v64, mean64), _one_pass(b64, v64), rtol=1e-12, atol=1e-12)

    b64, v64 = _spread_inputs(torch.float64, 300.0)
    reference = _one_pass(b64, v64)
    b32, v32 = b64.float(), v64.float()
    mean32 = torch.einsum("...hij,...jhs->...ihs", b32, v32)
    centered = (ro._centered_spread(b32, v32, mean32).double() - reference).abs().max()
    uncentered = (_one_pass(b32, v32).double() - reference).abs().max()
    assert centered < 1e-3 * uncentered, (float(centered), float(uncentered))


def test_centered_spread_tiles_carry_gradients(monkeypatch: pytest.MonkeyPatch) -> None:
    from vfe3.model import relational_operators as ro

    b, v = _spread_inputs(torch.float64, 2.0)
    v.requires_grad_(True)
    mean = torch.einsum("...hij,...jhs->...ihs", b, v)
    whole = torch.autograd.grad(ro._centered_spread(b, v, mean).square().sum(), v)[0]
    monkeypatch.setattr(ro, "_SPREAD_TILE_BYTES", 1)
    mean = torch.einsum("...hij,...jhs->...ihs", b, v)
    tiled = torch.autograd.grad(ro._centered_spread(b, v, mean).square().sum(), v)[0]
    torch.testing.assert_close(tiled, whole, rtol=1e-12, atol=1e-12)


# -- P2-19: routing support of the projected diagnostics ------------------------------------------

def test_relational_diagnostics_count_capped_keys_with_zero_beta() -> None:
    cfg = _cfg(_minimal_agent(predictive_readout="kernel", block_mlp_mode="coordinate"))
    torch.manual_seed(0)
    model = VFEModel(cfg)
    ids = torch.randint(0, V, (B, N), generator=torch.Generator().manual_seed(2))
    snapshot = model.build_diagnostic_snapshot(ids)
    capture = snapshot.layer_conditional_captures[-1]
    beta = torch.eye(N).expand_as(capture["beta_pre"]).clone()                 # collapsed rows, capped keys at 0
    score = torch.full_like(capture["score_energy_pre"], float(cfg.kl_max))
    score = score * (1.0 - torch.eye(N))                                       # every off-diagonal key capped
    capture["beta_pre"], capture["score_energy_pre"] = beta, score
    d = model._relational_diagnostics(snapshot)
    assert d["relational_score_klmax_frac"] == pytest.approx((N - 1) / (N + 1))   # causal support
    assert d["attn_entropy_min"] == 0.0


# -- hotspot-1#2: mean_distance tiles --------------------------------------------------------------

def test_mean_distance_tiles_match_the_direct_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    g = torch.Generator().manual_seed(5)
    a_q = torch.randn(2, 5, 2, 3, generator=g, dtype=torch.float64, requires_grad=True)
    a_k = torch.randn(2, 5, 2, 3, generator=g, dtype=torch.float64, requires_grad=True)
    direct = 0.5 * (a_q.movedim(-2, -3).unsqueeze(-2) - a_k.movedim(-2, -3).unsqueeze(-3)).square().sum(-1)
    monkeypatch.setattr(scoring, "_PROJECTION_TILE_BYTES", 1)
    tiled = scoring.projected_mean_distance_cost(a_q, None, a_k, None)
    torch.testing.assert_close(tiled, direct, rtol=0, atol=0)
    grads = torch.autograd.grad(tiled.sum(), (a_q, a_k))
    expected = torch.autograd.grad(direct.sum(), (a_q, a_k))
    for got, want in zip(grads, expected):
        torch.testing.assert_close(got, want)


# -- implementation-engineer#6: routing-cost registry ----------------------------------------------

def test_relational_score_dispatches_through_the_routing_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    import vfe3.gradients.relational as rel

    called = []
    for name in ("gaussian_kl", "noisy_query"):
        monkeypatch.setitem(scoring.ROUTING_COSTS, name,
                            lambda *args, _n=name, **kwargs: called.append(_n) or torch.zeros(()))
    eye = torch.eye(2, dtype=torch.float64).expand(1, 3, 2, 2, 2).clone()
    frame = CanonicalFrameContext(forward=eye, inverse=eye.clone(), block_diagonal=True)
    mu, sigma = torch.zeros(1, 3, 4, dtype=torch.float64), torch.ones(1, 3, 4, dtype=torch.float64)
    for routing in ("gaussian_kl", "noisy_query"):
        ops = make_relational_operators([2, 2], 11, readout_mode="predictive_content", routing=routing).double()
        rel.relational_score(mu, sigma, mu, sigma, frame, ops)
    assert called == ["gaussian_kl", "noisy_query"]
    with pytest.raises(ValueError, match="routing"):
        make_relational_operators([2, 2], 11, readout_mode="predictive_content", routing="dot_product")


# -- general-A#5 / hotspot-4#4: cached comparison RoPE ---------------------------------------------

def test_comparison_rope_rotation_is_cached_and_exact() -> None:
    from vfe3.geometry.rope import build_rope_rotation

    ops = VFEModel(_rope_cfg("comparison")).relational_operators
    first = ops.comparison_rope_rotation(N, 2, torch.device("cpu"), torch.float32)
    assert ops.comparison_rope_rotation(N, 2, torch.device("cpu"), torch.float32) is first
    expected = build_rope_rotation(torch.arange(N), [2], base=ops.comparison_rope_base,
                                   fraction=ops.comparison_rope_fraction, dtype=torch.float32)
    torch.testing.assert_close(first, expected, rtol=0, atol=0)
    with torch.inference_mode():
        inference = ops.comparison_rope_rotation(N + 1, 2, torch.device("cpu"), torch.float32)
    assert ops.comparison_rope_rotation(N + 1, 2, torch.device("cpu"), torch.float32) is not inference


# -- transformer-ml#4: rotated mean-field query covariance -----------------------------------------

def test_mean_field_rope_flags_a_rotated_query_covariance_that_loses_pd() -> None:
    from vfe3.geometry.rope import build_rope_rotation
    from vfe3.model.minimal_agent import minimal_agent_readout

    d, n = 4, 4
    R = build_rope_rotation(torch.arange(n), [d], base=10.0)
    g = torch.Generator().manual_seed(0)
    hit = None
    for _ in range(20000):
        Q = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))[0]
        smallest = 10 ** (-8.0 + 1.5 * torch.rand((), generator=g, dtype=torch.float64).item())
        C = ((Q * torch.tensor([1.0, 0.7, 0.3, smallest], dtype=torch.float64)) @ Q.T).float()
        C = 0.5 * (C + C.T)
        if int(torch.linalg.cholesky_ex(C)[1]) != 0:
            continue
        rotated = R[1:] @ C @ R[1:].transpose(-1, -2)
        if bool((torch.linalg.cholesky_ex(0.5 * (rotated + rotated.transpose(-1, -2)))[1] != 0).any()):
            hit = C
            break
    if hit is None:
        pytest.skip("no float32 covariance losing PD under rotation found for this seed")
    xi = torch.randn(1, n, d, generator=g)
    causal = torch.zeros(n, n).masked_fill(torch.triu(torch.ones(n, n, dtype=torch.bool), 1), float("-inf"))
    out = minimal_agent_readout(xi, hit.expand(1, n, d, d).clone(), torch.eye(d).unsqueeze(0),
                                torch.eye(d).unsqueeze(0), torch.full((1,), -2.0), causal, "mean_field",
                                check=False, rope=R)
    assert bool(out.failed)


# -- hotspot-1#3: float64 factor tiles under a query-only recovery ---------------------------------

def test_query_only_fp64_recovery_runs_the_factor_tiles_in_float64(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""The float64 fallback (the float32 shifted CholeskyQR3 retry reports failure) promotes both factors."""
    from tests.test_audit2_20261011_scorer import _failing_retry
    from tests.test_audit_20261008_fixes import _rank_ill_maps
    from vfe3.relational_scoring import projected_moments

    saved = scoring._CONDITION_RECOVERY
    scoring.set_projected_condition_recovery(True)
    monkeypatch.setattr(scoring, "cholesky_qr2", _failing_retry(scoring.cholesky_qr2))
    try:
        dtypes = []
        real = scoring._projected_kl_tile

        def spy(aq, ak, q_matrix, k_matrix, logdet_q, logdet_k, factor_trace):
            if factor_trace:
                dtypes.append((q_matrix.dtype, k_matrix.dtype))
            return real(aq, ak, q_matrix, k_matrix, logdet_q, logdet_k, factor_trace)

        monkeypatch.setattr(scoring, "_projected_kl_tile", spy)
        S_q, S_k = _rank_ill_maps(torch.float32)
        Hh, Kk, n = S_q.shape[0], S_q.shape[-1], 4
        d = Kk // Hh
        g = torch.Generator().manual_seed(1)
        mu = torch.randn(1, n, Kk, generator=g).unsqueeze(-2).expand(1, n, Hh, Kk)
        sigma = torch.ones(1, n, Kk).unsqueeze(-2).expand(1, n, Hh, Kk)
        inverse = torch.eye(d).expand(1, n, Hh, d, d).clone()
        a_q, G_q = projected_moments(mu, sigma, inverse, S_q)
        a_k, G_k = projected_moments(mu, sigma, inverse, S_k)
        scoring.projected_gaussian_kl(a_q, G_q, a_k, G_k, projection_q=(sigma, inverse, S_q, mu),
                                      projection_k=(sigma, inverse, S_k, mu))
    finally:
        scoring._CONDITION_RECOVERY = saved
    assert dtypes, "precondition: the query entries were recovered"
    assert all(pair == (torch.float64, torch.float64) for pair in dtypes), dtypes


# -- P2-21 / hotspot-3#3: IEEE fp32 pins ------------------------------------------------------------

def test_full_link_pullback_and_block_mlp_frame_action_run_at_ieee_fp32(
        monkeypatch: pytest.MonkeyPatch, _precision_high: None) -> None:
    from vfe3.gradients.full_link import _local_pullback
    from vfe3.model.block_mlp import _frame_action

    seen = _einsum_precisions(monkeypatch)
    eye = torch.eye(2).expand(1, 3, 2, 2, 2).clone()
    frame = CanonicalFrameContext(forward=eye, inverse=eye.clone(), block_diagonal=False)
    _local_pullback(torch.eye(4).expand(1, 3, 4, 4).clone(), torch.zeros(1, 3, 4),
                    CanonicalFrameContext(forward=eye, inverse=eye.clone(), block_diagonal=True))
    dense = torch.eye(4).expand(1, 3, 4, 4).clone()
    _frame_action(CanonicalFrameContext(forward=dense, inverse=dense.clone(), block_diagonal=False),
                  torch.zeros(1, 3, 4))(dense, torch.ones(1, 3, 4))
    assert seen and set(seen) == {"highest"}, seen
    assert torch.get_float32_matmul_precision() == "high"
    del frame


# -- hotspot-1#4: receiving-law retries have their own counter -------------------------------------

def test_receiving_law_retries_do_not_enter_the_projected_recovery_counter() -> None:
    from vfe3.gradients.full_link import _spd_precision, receiver_law_retry_counter

    generator = torch.Generator().manual_seed(1)
    M = torch.randn(6, 5, generator=generator, dtype=torch.float64)
    target = (M @ M.T + 1e-7 * torch.eye(6, dtype=torch.float64)).float()
    assert int(torch.linalg.cholesky_ex(target)[1]) != 0, "precondition: float32 factorization fails"
    recovery = scoring.source_recovery_counter(target.device)
    retries = receiver_law_retry_counter(target.device)
    _spd_precision(target)
    assert int(scoring.source_recovery_counter(target.device) - recovery) == 0
    assert int(receiver_law_retry_counter(target.device) - retries) == 1


# -- variational#3: one frozen functional for the receiver part deltas -----------------------------

def test_state_dependent_part_deltas_read_one_frozen_context() -> None:
    from tests.test_projected_gradient_receiver import _episode
    from vfe3.gradients.full_link import full_link_cost_parts

    belief, out, capture = _episode("channel_likelihood", "closed_form", n_iter=1,
                                    lambda_alpha_mode="state_dependent")
    context = capture["full_link_context"]                                     # frozen at the episode start
    for tag, state in (("before", belief), ("after", out)):
        mean_part, covariance_part, distance = full_link_cost_parts(state.mu, state.sigma, context)
        torch.testing.assert_close(capture[f"relational_receiver_mean_cost_{tag}"], mean_part.detach())
        torch.testing.assert_close(capture[f"relational_receiver_covariance_cost_{tag}"], covariance_part.detach())
        torch.testing.assert_close(capture[f"relational_receiver_target_distance_{tag}"], distance.detach())
    # The parts split the episode-initial majorizer, an upper bound on the profiled cost delta.
    part_delta = sum(float(capture[f"relational_receiver_{p}_cost_after"])
                     - float(capture[f"relational_receiver_{p}_cost_before"]) for p in ("mean", "covariance"))
    cost_delta = float(capture["relational_receiver_cost_after"]) - float(capture["relational_receiver_cost_before"])
    assert cost_delta <= part_delta + 1e-9
