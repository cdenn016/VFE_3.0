r"""Audit 2026-10-17 lane L1 (routing): items 2, 6, 9, 11 and 44.

Item 2: ``routing_kl_max_reference='query_shifted'`` clamps the cost without its query-only row constant, from
above only, so shrinking query variances no longer saturate the clamp and beta equals the absolute route's below
the cap. Item 6: the projection tile token cap grows with ``relational_tile_mib``. Item 9: the pre-clamp row
extremes, the diagonal-route term medians and the capped routing mass reach the diagnostics. Item 11: the diagonal
route has an O(N r) GEMM error estimate with a direct-tile fallback and admits ``pair_kl_gemm_tol``. Item 44: RoPE
angles are formed in float64. Tiny CPU models only (K <= 8, two heads, N <= 6, vocab <= 9); scorer probes at most
N = 64, r = 32, H = 2.
"""
import math
import warnings
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
import torch

import vfe3.relational_scoring as scoring
import vfe3.train as T
from tests.profiles import tiny_config
from vfe3.free_energy import attention_weights
from vfe3.geometry.rope import build_rope_rotation
from vfe3.gradients.relational import _diagonal_comparison_route, clamped_routing_cost
from vfe3.model.model import VFEModel
from vfe3.posthoc.klcap import KLCapObserver, causal_support

F64 = torch.float64
TOKENS = torch.tensor([[0, 1, 2, 3, 4, 5], [5, 6, 7, 8, 0, 1]])


def _causal_log_prior(n: int) -> torch.Tensor:                       # (N, N), -inf above the diagonal
    return torch.zeros(n, n).masked_fill(torch.ones(n, n, dtype=torch.bool).triu(1), -math.inf)


def _clamp(score: torch.Tensor, reference: str, kl_max: float, log_prior: torch.Tensor) -> torch.Tensor:
    return clamped_routing_cost(score, SimpleNamespace(kl_max_reference=reference), kl_max=kl_max,
                                log_prior=log_prior)


def _model(**overrides: Any) -> VFEModel:
    values: Dict[str, Any] = dict(vocab_size=9, embed_dim=8, n_heads=2, max_seq_len=6, n_layers=2, phi_scale=0.4)
    values.update(overrides)
    torch.manual_seed(3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VFEModel(tiny_config(**values)).eval()


# ------------------------------------------------------------------------------------------------ item 2

@pytest.mark.parametrize("route", ["diagonal", "dense"])
def test_dropped_query_constant_is_the_cost_plus_half_r_plus_logdet(route: str) -> None:
    g = torch.Generator().manual_seed(1)
    B, N, H, r = 2, 5, 2, 4
    a_q, a_k = (torch.randn(B, N, H, r, generator=g, dtype=F64) for _ in range(2))
    g_q, g_k = (0.3 + torch.rand(B, N, H, r, generator=g, dtype=F64) for _ in range(2))
    if route == "diagonal":
        args, options = (a_q, g_q, a_k, g_k), {"diagonal_covariance": True}
    else:
        args, options = (a_q, torch.diag_embed(g_q), a_k, torch.diag_embed(g_k)), {"symmetric_inputs": True}
    D = scoring.projected_gaussian_kl(*args, **options)                       # (B, H, N, N)
    shifted = scoring.projected_gaussian_kl(*args, drop_query_constant=True, **options)
    constant = -.5 * (r + g_q.log().sum(-1)).movedim(-1, -2).unsqueeze(-1)    # (B, H, N, 1) c_i
    torch.testing.assert_close(shifted, D - constant, rtol=0, atol=1e-12)


def test_noisy_query_drops_its_query_logdet() -> None:
    g = torch.Generator().manual_seed(2)
    B, N, H, r = 1, 4, 2, 3
    a_q, a_k = (torch.randn(B, N, H, r, generator=g, dtype=F64) for _ in range(2))
    root = torch.randn(2, B, N, H, r, r, generator=g, dtype=F64)
    G_q, G_k = (x @ x.transpose(-1, -2) + 0.5 * torch.eye(r, dtype=F64) for x in root)
    D = scoring.projected_noisy_query_cost(a_q, G_q, a_k, G_k)
    shifted = scoring.projected_noisy_query_cost(a_q, G_q, a_k, G_k, drop_query_constant=True)
    constant = -.5 * torch.logdet(G_q).movedim(-1, -2).unsqueeze(-1)
    torch.testing.assert_close(shifted, D - constant, rtol=0, atol=1e-12)


def test_query_shifted_does_not_saturate_on_shrunk_query_variances() -> None:
    # The audit's probe: every query variance shrunk by 1e-4 clamps every row under the absolute reference.
    g = torch.Generator().manual_seed(0)
    B, N, H, r = 1, 16, 2, 8
    kl_max = 4.0 * r
    a = torch.randn(B, N, H, r, generator=g)
    var = 0.5 + torch.rand(B, N, H, r, generator=g)
    log_prior = _causal_log_prior(N)
    support = torch.isfinite(log_prior)
    scored = support.sum(-1) >= 2                                              # (N,)
    absolute = _clamp(scoring.projected_gaussian_kl(a, 1e-4 * var, a, var, diagonal_covariance=True),
                      "absolute", kl_max, log_prior)
    shifted = _clamp(scoring.projected_gaussian_kl(a, 1e-4 * var, a, var, diagonal_covariance=True,
                                                   drop_query_constant=True),
                     "query_shifted", kl_max, log_prior)

    def fully_clamped(score: torch.Tensor) -> int:
        capped = (score >= kl_max) & support
        return int(((capped.sum(-1) == support.sum(-1)) & scored).sum())

    assert fully_clamped(absolute) == H * int(scored.sum())                   # every scored row saturates
    assert fully_clamped(shifted) < H * int(scored.sum())
    assert fully_clamped(shifted) == 0
    # the absolute router then routes by the prior alone; the shifted one keeps its content dependence
    beta_abs = attention_weights(absolute, log_prior=log_prior)
    torch.testing.assert_close(beta_abs, attention_weights(torch.zeros_like(absolute), log_prior=log_prior))
    beta_shift = attention_weights(shifted, log_prior=log_prior)
    assert float((beta_shift - beta_abs).abs().max()) > 1e-2


def test_query_shifted_beta_equals_absolute_below_the_cap() -> None:
    g = torch.Generator().manual_seed(4)
    B, N, H, r = 2, 6, 2, 4
    a_q, a_k = (torch.randn(B, N, H, r, generator=g, dtype=F64) for _ in range(2))
    g_q, g_k = (0.4 + torch.rand(B, N, H, r, generator=g, dtype=F64) for _ in range(2))
    log_prior = _causal_log_prior(N)
    kl_max = 1e4
    D = scoring.projected_gaussian_kl(a_q, g_q, a_k, g_k, diagonal_covariance=True)
    shifted = scoring.projected_gaussian_kl(a_q, g_q, a_k, g_k, diagonal_covariance=True, drop_query_constant=True)
    assert float(D.min()) >= -1e-12 and float(shifted.max()) < kl_max         # nothing clamps on either route
    beta_abs = attention_weights(_clamp(D, "absolute", kl_max, log_prior), tau=2.0, log_prior=log_prior)
    beta_shift = attention_weights(_clamp(shifted, "query_shifted", kl_max, log_prior), tau=2.0, log_prior=log_prior)
    torch.testing.assert_close(beta_shift, beta_abs, rtol=0, atol=1e-13)


def test_query_shifted_clamps_from_above_only() -> None:
    score = torch.tensor([[-3.0, 0.5, 9.0, float("nan"), float("inf")]])
    out = _clamp(score, "query_shifted", 2.0, None)
    assert out.tolist() == [[-3.0, 0.5, 2.0, 2.0, 2.0]]


def test_config_accepts_query_shifted_and_rejects_unknown_references() -> None:
    assert tiny_config(routing_kl_max_reference="query_shifted").routing_kl_max_reference == "query_shifted"
    with pytest.raises(ValueError, match="routing_kl_max_reference"):
        tiny_config(routing_kl_max_reference="query")


@pytest.mark.parametrize("rope_mode", ["none", "comparison_invariant"])
def test_model_beta_matches_absolute_when_nothing_clamps(rope_mode: str) -> None:
    common = dict(kl_max=1e6, rope_mode=rope_mode)
    absolute = _model(routing_kl_max_reference="absolute", **common)
    shifted = _model(routing_kl_max_reference="query_shifted", **common)
    shifted.load_state_dict(absolute.state_dict())
    snap_abs = absolute.build_diagnostic_snapshot(TOKENS)
    snap_shift = shifted.build_diagnostic_snapshot(TOKENS)
    for layer in range(2):
        torch.testing.assert_close(shifted._layer_capture(snap_shift, layer)["beta_pre"],
                                   absolute._layer_capture(snap_abs, layer)["beta_pre"], rtol=0, atol=2e-6)
    torch.testing.assert_close(snap_shift.logits, snap_abs.logits, rtol=0, atol=2e-5)
    # the router still receives a gradient through the shifted cost
    shifted.train()
    logits = shifted(TOKENS)
    logits = logits[0] if isinstance(logits, tuple) else logits
    logits.float().square().mean().backward()
    grads = [p.grad for name, p in shifted.named_parameters() if "raw_score" in name]
    assert grads and all(grad is not None and bool(torch.isfinite(grad).all()) for grad in grads)
    assert any(float(grad.abs().max()) > 0 for grad in grads)


# ------------------------------------------------------------------------------------------------ item 11

def _displaced_probe(displacement: float, seed: int = 1) -> List[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    N, H, r = 64, 2, 32
    a = torch.randn(1, N, H, r, generator=g, dtype=F64)
    a[:, 0] += displacement                                                    # the first key, displaced by D
    var = torch.full((1, N, H, r), 0.5, dtype=F64)
    return [a.float(), var.float()]


@pytest.mark.parametrize("displacement", [1e3, 1e4])
def test_diagonal_fallback_rescores_the_displaced_first_key_probe(displacement: float) -> None:
    a, var = _displaced_probe(displacement)
    exact = scoring.projected_gaussian_kl(a.double(), var.double(), a.double(), var.double(),
                                          diagonal_covariance=True)            # float64 of the float32 inputs
    relevant = exact < 128.0                                                   # pairs below the live cap
    plain = scoring.projected_gaussian_kl(a, var, a, var, diagonal_covariance=True).double()
    scoring.reset_gemm_fallback_slices()
    rescored = scoring.projected_gaussian_kl(a, var, a, var, diagonal_covariance=True,
                                             gemm_expansion_tol=1e-3).double()
    assert scoring.gemm_fallback_slices() == 2                                 # both heads flagged
    assert float((plain - exact)[relevant].abs().max()) > 0.5                  # the GEMM's error on this probe
    assert float((rescored - exact)[relevant].abs().max()) < 1e-3
    estimate = scoring._diagonal_gemm_expansion_error(a.movedim(-2, -3), a.movedim(-2, -3), var.movedim(-2, -3))
    assert float(estimate.min()) > float((plain - exact)[relevant].abs().max())   # the estimate majorizes it


def test_diagonal_fallback_is_inert_on_a_healthy_slice_and_differentiable() -> None:
    a, var = _displaced_probe(0.0, seed=2)
    scoring.reset_gemm_fallback_slices()
    plain = scoring.projected_gaussian_kl(a, var, a, var, diagonal_covariance=True)
    gated = scoring.projected_gaussian_kl(a, var, a, var, diagonal_covariance=True, gemm_expansion_tol=1e-3)
    assert scoring.gemm_fallback_slices() == 0
    assert torch.equal(plain, gated)
    # a flagged slice's direct tiles carry the gradient
    a_big, var_big = _displaced_probe(1e3, seed=3)
    leaves = [x[:, :8].clone().requires_grad_() for x in (a_big, var_big)]
    out = scoring.projected_gaussian_kl(leaves[0], leaves[1], leaves[0], leaves[1], diagonal_covariance=True,
                                        gemm_expansion_tol=1e-6, drop_query_constant=True)
    out.sum().backward()
    assert all(leaf.grad is not None and bool(torch.isfinite(leaf.grad).all()) for leaf in leaves)


def test_router_keeps_the_diagonal_route_with_a_gemm_tolerance() -> None:
    operators = SimpleNamespace(routing="gaussian_kl", pair_kl_gemm_tol=1e-3)
    assert _diagonal_comparison_route(operators, 4, 2)
    model = _model(rope_mode="comparison_invariant", pair_kl_gemm_tol=1e-3)
    reference = _model(rope_mode="comparison_invariant")
    model.load_state_dict(reference.state_dict())
    scoring.reset_gemm_fallback_slices()
    snap, snap_ref = model.build_diagnostic_snapshot(TOKENS), reference.build_diagnostic_snapshot(TOKENS)
    torch.testing.assert_close(snap.logits, snap_ref.logits, rtol=0, atol=0)
    assert "score_term_medians_pre" in model._layer_capture(snap, 0)           # the diagonal route ran


# ------------------------------------------------------------------------------------------------ item 6

def test_tile_token_cap_follows_the_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    assert scoring._tile_token_cap() == 32                                     # the default budget
    monkeypatch.setattr(scoring, "_PROJECTION_TILE_BYTES", 256 * 1024 ** 2)
    assert scoring._tile_token_cap() == 256
    monkeypatch.setattr(scoring, "_PROJECTION_TILE_BYTES", 1024 ** 2)
    assert scoring._tile_token_cap() == 32


def test_raised_budget_dispatches_fewer_projection_tiles(monkeypatch: pytest.MonkeyPatch) -> None:
    g = torch.Generator().manual_seed(7)
    N, H, d, r = 100, 1, 2, 2
    mu = torch.randn(1, N, H, d, generator=g)
    sigma = 0.5 + torch.rand(1, N, H, d, generator=g)
    inverse = torch.eye(d).expand(1, N, H, d, d)
    S = torch.randn(H, r, d, generator=g)
    full = scoring._projected_moments_tile(mu, sigma, inverse, S)
    calls: List[int] = []
    original = scoring._projected_moments_tile

    def counting(*args: Any) -> Any:
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(scoring, "_projected_moments_tile", counting)
    monkeypatch.setattr(scoring, "_CHECKPOINT_BYTES", 1024)                    # 16 bytes per token: tiles
    monkeypatch.setattr(scoring, "_PROJECTION_TILE_BYTES", 1024)              # cap 32: four tiles of 32 tokens
    small = scoring.projected_moments(mu, sigma, inverse, S)
    assert len(calls) == 4
    calls.clear()
    monkeypatch.setattr(scoring, "_PROJECTION_TILE_BYTES", 4 * 1024)          # cap 128: one tile
    large = scoring.projected_moments(mu, sigma, inverse, S)
    assert len(calls) == 1
    for result in (small, large):
        torch.testing.assert_close(result[0], full[0], rtol=0, atol=0)
        torch.testing.assert_close(result[1], full[1], rtol=0, atol=0)


# ------------------------------------------------------------------------------------------------ item 44

def test_rope_relative_position_holds_at_the_float32_floor() -> None:
    N, r, t = 4096, 32, 5
    R = build_rope_rotation(torch.arange(N), [r], base=100.0, dtype=torch.float32)
    assert R.dtype == torch.float32
    R64 = R.double()
    product = R64[:-t].transpose(-1, -2) @ R64[t:]                             # R_i^T R_{i+t}, (N - t, r, r)
    defect = float((product - R64[t]).abs().max())
    assert defect < 5e-7                                                       # fp32 angles gave 1.7e-4 here
    exact = build_rope_rotation(torch.arange(N), [r], base=100.0, dtype=F64)
    assert float((R64 - exact).abs().max()) <= 2 ** -24 + 1e-12                # one rounding of the float64 entries


# ------------------------------------------------------------------------------------------------ item 9

@pytest.mark.parametrize("rope_mode", ["none", "comparison_invariant"])
def test_diagnostics_report_preclamp_costs_and_capped_mass(rope_mode: str) -> None:
    model = _model(rope_mode=rope_mode, kl_max=6.0)
    snapshot = model.build_diagnostic_snapshot(TOKENS)
    d = model.diagnostics(TOKENS, snapshot=snapshot)
    per_layer = model.diagnostics_per_layer(TOKENS, snapshot=snapshot)
    names = ([f"relational_raw_row_{name}_{level}" for name in ("min", "gap") for level in ("q10", "q50", "q90")]
             + ["relational_capped_mass_mean", "relational_capped_mass_max"]
             + [f"relational_score_term_{term}_median"
                for term in ("query_constant", "key_logdet", "trace", "mahalanobis")])
    guard = ["relational_raw_row_min_median_head_max", "relational_raw_row_min_median_argmax_layer",
             "relational_raw_row_min_median_argmax_head", "relational_capped_mass_mean_head_max"]
    val_keys = set(T._VAL_RELATIONAL_DIAG_KEYS)
    for key in names + guard:
        assert key in d and f"val_{key}" in val_keys, key
    diagonal = rope_mode == "comparison_invariant"
    for layer in range(2):
        cap = model._layer_capture(snapshot, layer)
        beta, score = cap["beta_pre"], cap["score_energy_pre"]
        support = causal_support(beta, snapshot.log_prior)
        rows = support.sum(-1) >= 2
        low = cap["score_raw_row_min_pre"]                                     # (B, H, N)
        high = cap["score_raw_row_max_pre"]
        assert low.shape == rows.shape and bool((high >= low)[rows].all())
        # the clamped cost under the absolute reference is the raw cost clamped, so the raw minimum bounds it
        clamped_min = score.masked_fill(~support, math.inf).amin(-1)
        torch.testing.assert_close(clamped_min[rows].double(), low[rows].clamp(0, model.cfg.kl_max),
                                   rtol=1e-6, atol=1e-6)
        expected_q50 = float(torch.quantile(low[rows], 0.5))
        assert per_layer["relational_raw_row_min_q50"][layer] == pytest.approx(expected_q50, rel=1e-12)
        capped = support & rows.unsqueeze(-1) & (score >= model.cfg.kl_max)
        mass = (beta.double() * capped).sum(-1)[rows]
        assert per_layer["relational_capped_mass_mean"][layer] == pytest.approx(float(mass.mean()), abs=1e-12)
        assert per_layer["relational_capped_mass_max"][layer] == pytest.approx(float(mass.max()), abs=1e-12)
        medians = [per_layer[f"relational_score_term_{term}_median"][layer]
                   for term in ("query_constant", "key_logdet", "trace", "mahalanobis")]
        assert all(math.isfinite(v) for v in medians) if diagonal else all(math.isnan(v) for v in medians)
        if diagonal:
            assert cap["score_term_medians_pre"].shape == (4, 3)
            assert medians[2] > 0 and medians[3] >= 0                           # trace and Mahalanobis terms
    assert d["relational_capped_mass_mean_head_max"] >= max(per_layer["relational_capped_mass_mean"]) - 1e-12


def test_term_medians_reproduce_the_cost_terms() -> None:
    g = torch.Generator().manual_seed(9)
    B, N, H, r = 1, 4, 2, 4
    a_q, a_k = (torch.randn(B, N, H, r, generator=g, dtype=F64) for _ in range(2))
    g_q, g_k = (0.3 + torch.rand(B, N, H, r, generator=g, dtype=F64) for _ in range(2))
    medians = scoring.diagonal_kl_term_medians(a_q, g_q, a_k, g_k)            # (4, H + 1)
    aq, ak, gq, gk = (x.movedim(-2, -3) for x in (a_q, a_k, g_q, g_k))         # (B, H, N, r)
    query = (-.5 * (r + gq.log().sum(-1))).unsqueeze(-1).expand(B, H, N, N)
    key = (.5 * gk.log().sum(-1)).unsqueeze(-2).expand(B, H, N, N)
    trace = .5 * (gq.unsqueeze(-2) / gk.unsqueeze(-3)).sum(-1)
    maha = .5 * ((aq.unsqueeze(-2) - ak.unsqueeze(-3)).square() / gk.unsqueeze(-3)).sum(-1)
    D = scoring.projected_gaussian_kl(a_q, g_q, a_k, g_k, diagonal_covariance=True)
    torch.testing.assert_close(query + key + trace + maha, D, rtol=0, atol=1e-12)
    for row, term in enumerate((query, key, trace, maha)):
        for head in range(H):
            assert float(medians[row, head]) == pytest.approx(float(term[:, head].reshape(-1).median()), rel=1e-12)
        assert float(medians[row, H]) == pytest.approx(float(term.reshape(-1).median()), rel=1e-12)


def test_klcap_observer_reads_the_preclamp_capture() -> None:
    model = _model(kl_max=6.0)
    snapshot = model.build_diagnostic_snapshot(TOKENS)
    captures = [model._layer_capture(snapshot, layer) for layer in range(2)]
    observer = KLCapObserver(model.cfg.kl_max)
    valid = torch.ones(TOKENS.shape, dtype=torch.bool)
    observer.update({"layer_conditional_captures": captures, "log_prior": snapshot.log_prior}, valid,
                    {"windows": [0, 1]})
    rows = observer.rows()
    assert rows and all(math.isfinite(row["raw_row_min_mean"]) and math.isfinite(row["raw_row_gap_mean"])
                        for row in rows)
    for row in rows:
        cap = captures[row["layer"]]
        support = causal_support(cap["beta_pre"], snapshot.log_prior)
        scored = (support.sum(-1) >= 2)[:, row["head"]]
        low = cap["score_raw_row_min_pre"][:, row["head"]][scored]
        assert row["raw_row_min_mean"] == pytest.approx(float(low.mean()), rel=1e-12)
        assert row["raw_row_min_max"] == pytest.approx(float(low.max()), rel=1e-12)
        assert 0.0 <= row["capped_beta_mass_max"] <= 1.0 + 1e-6
