r"""Audit 2026-10-31 remediation, package ``router``: the kernel router's clamp chain, its causal tiles and the
full-link receiver.

* A138: the query-shifted row reference holds a boolean mask for backward where it held a float weight grid, and the
  router cap and :func:`vfe3.families.base.safe_kl_clamp` are one autograd Function that saves a boolean mask. Pinned
  bitwise against verbatim copies of the 5bb0a57 chain at the unit level (with injected NaN and :math:`+\infty`
  costs), on the loss and every parameter gradient of a tiny copy of the live route, and by a saved-tensor census.
* A155: the finite mask of an attention log-prior is computed once per prior tensor and version.
* A140: the opt-in ``routing_causal_tiles`` agrees with the full grid to rounding, forms the causal fraction of it,
  and leaves the full grid bitwise unchanged when off or under a prior that is not strictly causal.
* A193: the full-link closed-form episode forms no unread force, reuses the built context on its first iteration and
  factors a full prior covariance once.
* A239: the noisy-key scale symmetry of the query-shifted cost carries the key noise, :math:`n_h \to c^2n_h`.
* A255: the noisy-key diagonal route records the cost-term medians of its broadened key law :math:`g_j + n_h`.
* A275: the ALiBi builders refuse the misspelled keyword ``slope``.
* A334: the receiving-law float64 retry count reaches :meth:`vfe3.model.model.VFEModel.diagnostics`.
* A344: ``_transport`` returns the dense Omega without ``right_phi`` whatever ``materialize`` is.

Configurations come from :mod:`tests.profiles` (the frozen K = 8 reference profile with the live route's switches of
``docs/audits/ultradeep-2026-10-31/live-config.json``); tiny CPU models only.
"""
from __future__ import annotations

import math
import warnings
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest
import torch

import vfe3.families.base as fam_base
import vfe3.families.frame_gaussian as fam_frame
import vfe3.families.gaussian as fam_gaussian
import vfe3.gradients.full_link as full_link
import vfe3.gradients.relational as gr
import vfe3.inference.e_step as e_step_module
import vfe3.model.relational_operators as ro
import vfe3.relational_scoring as rs
import vfe3.train as T
from tests.profiles import make_relational_operators, reference_k8_values, test_config
from tests.test_full_link_gradients import _fixture
from vfe3.attention_prior import attention_log_prior
from vfe3.belief import BeliefState
from vfe3.config import VFE3Config, config_from_serialized, serialize_config
from vfe3.contracts import CanonicalFrameContext
from vfe3.free_energy import attention_weights
from vfe3.geometry.transport import FactoredTransport
from vfe3.model.model import VFEModel

F64 = torch.float64
B, N, VOCAB = 2, 6, 9
TOKENS = torch.tensor([[0, 1, 2, 3, 4, 5], [5, 6, 7, 8, 1, 3]])
TARGETS = torch.tensor([[1, 2, 3, 4, 5, 6], [6, 7, 8, 0, 2, 4]])
REFERENCES = ("query_shifted", "query_shifted_v1", "row_min", "absolute")


@pytest.fixture(autouse=True)
def _two_threads_quiet() -> Any:
    previous = torch.get_num_threads()
    torch.set_num_threads(min(2, previous))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield
    torch.set_num_threads(previous)


# ------------------------------------------------------------------------------------------- the 5bb0a57 chain

def _legacy_safe_kl_clamp(kl: torch.Tensor, *, kl_max: float = 100.0) -> torch.Tensor:
    r"""``safe_kl_clamp`` as of 5bb0a57."""
    kl = kl.clamp(min=0.0, max=kl_max)
    return kl.nan_to_num(nan=kl_max, posinf=kl_max, neginf=0.0)


def _legacy_row_reference(score: torch.Tensor, key_log_volume: torch.Tensor,
                          log_prior: Optional[torch.Tensor] = None) -> torch.Tensor:
    r"""``query_shifted_row_reference`` as of 5bb0a57 (float admissibility weights)."""
    finite_volume = torch.isfinite(key_log_volume).unsqueeze(-2)
    admissible = torch.isfinite(score) & finite_volume
    if log_prior is not None:
        admissible = admissible & torch.isfinite(log_prior)
    volume = torch.where(finite_volume, key_log_volume.unsqueeze(-2),
                         torch.zeros_like(key_log_volume.unsqueeze(-2)))
    weights = admissible.to(score.dtype)
    count = weights.sum(-1, keepdim=True)
    return (weights * volume.to(score.dtype)).sum(-1, keepdim=True) / count.clamp(min=1)


def _legacy_clamped_routing_cost(score: torch.Tensor, operators: Any, *, kl_max: float,
                                 log_prior: Optional[torch.Tensor] = None,
                                 key_log_volume: Optional[torch.Tensor] = None) -> torch.Tensor:
    r"""``clamped_routing_cost`` as of 5bb0a57."""
    if operators.kl_max_reference == "query_shifted_v1":
        return score.clamp(max=kl_max).nan_to_num(nan=kl_max, posinf=kl_max)
    if operators.kl_max_reference == "query_shifted":
        shifted = score - _legacy_row_reference(score, key_log_volume, log_prior)
        return shifted.clamp(max=kl_max).nan_to_num(nan=kl_max, posinf=kl_max)
    if operators.kl_max_reference == "row_min":
        admissible = torch.isfinite(score)
        if log_prior is not None:
            admissible = admissible & torch.isfinite(log_prior)
        row_min = score.masked_fill(~admissible, float("inf")).amin(dim=-1, keepdim=True)
        row_min = torch.where(torch.isfinite(row_min), row_min, torch.zeros_like(row_min))
        score = score - row_min
    return _legacy_safe_kl_clamp(score, kl_max=kl_max)


def _install_legacy_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""Route every caller of the router clamp and of ``safe_kl_clamp`` through the 5bb0a57 copies."""
    monkeypatch.setattr(gr, "clamped_routing_cost", _legacy_clamped_routing_cost)
    monkeypatch.setattr(full_link, "clamped_routing_cost", _legacy_clamped_routing_cost)
    for module in (fam_base, fam_gaussian, fam_frame):
        monkeypatch.setattr(module, "safe_kl_clamp", _legacy_safe_kl_clamp)


# ------------------------------------------------------------------------------------------- fixtures

def _live_values(**overrides: Any) -> Dict[str, Any]:
    r"""The live route's switches on the frozen K = 8 reference profile, two heads of width 4 (no launcher read)."""
    values: Dict[str, Any] = dict(
        relational_routing="noisy_key", beta_attention_prior="causal_alibi", alibi_slope=1.0,
        predictive_covariance="passthrough", routing_kl_max_reference="query_shifted",
        rope_mode="comparison_invariant", rope_base=100.0, rope_fraction=1.0, projected_factorization="gram",
        family="gaussian_diagonal", gauge_group="block_glk", decode_mode="diagonal_chunked",
        norm_type_block="layernorm", use_block_mlp=True, block_mlp_mode="coordinate",
        block_mlp_covariance="passthrough", pos_phi="none", vocab_size=VOCAB, max_seq_len=N, n_layers=2,
        learn_attention_temperature=False)
    values.update(overrides)
    return reference_k8_values(**values)


def _model(dtype: torch.dtype = torch.float32, **overrides: Any) -> VFEModel:
    torch.manual_seed(7)
    model = VFEModel(VFE3Config(**_live_values(**overrides)))
    gen = torch.Generator().manual_seed(1)
    with torch.no_grad():                                       # generic frames and tables, away from the identity
        model.prior_bank.phi_embed.copy_(0.3 * torch.randn(model.prior_bank.phi_embed.shape, generator=gen))
        model.prior_bank.mu_embed.copy_(torch.randn(model.prior_bank.mu_embed.shape, generator=gen))
    return model.to(dtype)


def _loss_and_grads(model: VFEModel) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    model.zero_grad(set_to_none=True)
    _, loss, _ = model(TOKENS, TARGETS)
    loss.backward()
    return loss.detach(), {name: p.grad.detach().clone() for name, p in model.named_parameters()
                           if p.grad is not None}


def _assert_bitwise(new: Tuple[torch.Tensor, Dict[str, torch.Tensor]],
                    old: Tuple[torch.Tensor, Dict[str, torch.Tensor]]) -> None:
    assert torch.equal(new[0], old[0])
    assert new[1].keys() == old[1].keys() and len(old[1]) > 0
    for name in old[1]:
        assert torch.equal(new[1][name], old[1][name]), name


def _assert_close_scaled(a: torch.Tensor, b: torch.Tensor, rtol: float, atol: float, msg: str = "") -> None:
    r"""Elementwise agreement with the absolute tolerance scaled by the reference's largest magnitude."""
    scale = float(b.abs().max()) if b.numel() else 1.0
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol * max(scale, 1e-30), msg=msg or None)


def _counting(monkeypatch: pytest.MonkeyPatch, module: Any, name: str) -> Dict[str, int]:
    calls = {"n": 0}
    original = getattr(module, name)

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        return original(*args, **kwargs)
    monkeypatch.setattr(module, name, wrapped)
    return calls


def _operators(dtype: torch.dtype = F64, **overrides: Any) -> ro.RelationalOperators:
    r"""A kernel router on the diagonal comparison route: noisy-key, query-shifted, comparison-invariant RoPE."""
    values: Dict[str, Any] = dict(readout_mode="predictive_content", attention_residual=True,
                                  predictive_covariance="passthrough", routing="noisy_key",
                                  content_link_chart="exponential", kl_max_reference="query_shifted",
                                  rope_comparison_covariance="rotation_invariant", comparison_rope_base=100.0,
                                  comparison_rope_fraction=1.0, routing_key_noise=1.0)
    values.update(overrides)
    torch.manual_seed(3)
    return make_relational_operators([4, 4], 5, **values).to(dtype)


def _belief(n: int = N, dtype: torch.dtype = F64, seed: int = 2) -> Tuple[torch.Tensor, ...]:
    r"""Means (B, n, 8), variances (B, n, 8) and a compact block frame (B, n, 2, 4, 4) away from the identity."""
    g = torch.Generator().manual_seed(seed)
    U = torch.linalg.matrix_exp(.3 * torch.randn(B, n, 2, 4, 4, generator=g, dtype=dtype))
    mu = torch.randn(B, n, 8, generator=g, dtype=dtype)
    sigma = .5 + torch.rand(B, n, 8, generator=g, dtype=dtype)
    return mu, sigma, CanonicalFrameContext(U, torch.linalg.inv(U), True)


@pytest.fixture()
def _diagonal_route(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""The diagonal comparison route needs conditioning-triggered recovery off (projected_factorization='gram')."""
    monkeypatch.setattr(rs, "_CONDITION_RECOVERY", False)


# ------------------------------------------------------------------------------------------- A138: unit level

@pytest.mark.parametrize("lower, upper, neginf", [(None, 3.0, None), (0.0, 3.0, 0.0), (0.0, math.inf, 0.0),
                                                  (None, math.inf, None)])
@pytest.mark.parametrize("dtype", [torch.float32, F64])
def test_clamp_nonfinite_is_the_composite_in_value_and_gradient(lower: Optional[float], upper: float,
                                                                neginf: Optional[float], dtype: torch.dtype) -> None:
    g = torch.Generator().manual_seed(0)
    x0 = 4.0 * torch.randn(5, 7, generator=g, dtype=dtype)
    x0[0, 0], x0[0, 1], x0[0, 2], x0[1, 3] = float("nan"), float("inf"), float("-inf"), 3.0     # the cap itself
    x0[2, 4] = 0.0                                                                              # the floor itself
    cotangent = torch.randn(5, 7, generator=g, dtype=dtype)
    x_new, x_old = (x0.clone().requires_grad_() for _ in range(2))
    new = fam_base.clamp_nonfinite(x_new, lower=lower, upper=upper, neginf=neginf)
    old = x_old.clamp(min=lower, max=upper).nan_to_num(nan=upper, posinf=upper, neginf=neginf)
    assert new.dtype == old.dtype and torch.equal(new, old)
    (new * cotangent).sum().backward()
    (old * cotangent).sum().backward()
    assert torch.equal(x_new.grad, x_old.grad)
    with torch.no_grad():                                                   # the composite without autograd
        assert torch.equal(fam_base.clamp_nonfinite(x0, lower=lower, upper=upper, neginf=neginf), old.detach())


def test_clamp_nonfinite_double_backward_matches_the_composite() -> None:
    x0 = torch.tensor([-1.0, 0.5, 2.0, 5.0], dtype=F64)
    results = []
    for function in (lambda x: fam_base.clamp_nonfinite(x, lower=0.0, upper=3.0, neginf=0.0),
                     lambda x: _legacy_safe_kl_clamp(x, kl_max=3.0)):
        x = x0.clone().requires_grad_()
        v = torch.ones_like(x, requires_grad=True)
        (grad,) = torch.autograd.grad(function(x), x, v, create_graph=True)
        (second,) = torch.autograd.grad(grad.square().sum(), v)
        results.append((grad.detach(), second))
    assert all(torch.equal(a, b) for a, b in zip(*results))


def _router_operands(dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, ...]:
    r"""A (B, H, N, N) cost grid with an injected NaN and +inf, (B, H, N) key log-volumes with a NaN and a causal
    ALiBi prior."""
    g = torch.Generator().manual_seed(4)
    score = 3.0 * torch.randn(B, 2, N, N, generator=g, dtype=dtype)
    score[0, 1, 3, 1], score[1, 0, 4, 4] = float("nan"), float("inf")
    volume = torch.randn(B, 2, N, generator=g, dtype=dtype)
    volume[1, 1, 2] = float("nan")
    prior = attention_log_prior("causal_alibi", N, N, n_heads=2, alibi_slope=1.0, dtype=dtype)
    return score, volume, prior


@pytest.mark.parametrize("reference", REFERENCES)
@pytest.mark.parametrize("kl_max", [1.5, 128.0])
def test_router_clamp_chain_is_bitwise_the_legacy_chain(reference: str, kl_max: float) -> None:
    score0, volume0, prior = _router_operands()
    ops = SimpleNamespace(kl_max_reference=reference)
    g = torch.Generator().manual_seed(5)
    cotangent = torch.randn(B, 2, N, N, generator=g)
    runs = []
    for clamp in (gr.clamped_routing_cost, _legacy_clamped_routing_cost):
        score, volume = score0.clone().requires_grad_(), volume0.clone().requires_grad_()
        clamped = clamp(score, ops, kl_max=kl_max, log_prior=prior, key_log_volume=volume)
        beta = attention_weights(clamped, tau=1.0, log_prior=prior)
        (beta * cotangent).sum().backward()
        runs.append((clamped.detach(), beta.detach(), score.grad, volume.grad))
    if kl_max == 1.5 and reference != "absolute":
        assert bool((runs[0][0] == kl_max).any())                             # the cap binds
    for new, old in zip(*runs):
        assert (new is None) == (old is None)
        if new is not None:
            assert torch.equal(new, old)


def test_query_shifted_row_reference_is_bitwise_the_weighted_mean() -> None:
    score, volume0, prior = _router_operands(F64)
    g = torch.Generator().manual_seed(6)
    cotangent = torch.randn(B, 2, N, 1, generator=g, dtype=F64)
    runs = []
    for reference in (gr.query_shifted_row_reference, _legacy_row_reference):
        volume = volume0.clone().requires_grad_()
        m = reference(score, volume, prior)
        (m * cotangent).sum().backward()
        runs.append((m.detach(), volume.grad))
    assert torch.equal(runs[0][0], runs[1][0]) and torch.equal(runs[0][1], runs[1][1])


def _saved_floats(function: Callable[[], torch.Tensor], numel: int) -> Tuple[int, int]:
    r"""(float, bool) tensors of ``numel`` elements that autograd packs for backward while ``function`` runs."""
    saved: List[torch.Tensor] = []

    def pack(t: torch.Tensor) -> torch.Tensor:
        saved.append(t)
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        function()
    grids = [t for t in saved if t.numel() == numel]
    return (sum(t.dtype.is_floating_point for t in grids), sum(t.dtype == torch.bool for t in grids))


def test_saved_tensor_census_holds_no_float_pair_grid() -> None:
    score0, volume0, prior = _router_operands()
    score0 = score0.nan_to_num(0.0, 0.0)                                     # leaves as the router sees them
    ops = SimpleNamespace(kl_max_reference="query_shifted")

    def run(clamp: Callable[..., torch.Tensor]) -> Callable[[], torch.Tensor]:
        score, volume = score0.clone().requires_grad_(), volume0.clone().requires_grad_()
        return lambda: clamp(score, ops, kl_max=1.5, log_prior=prior, key_log_volume=volume)

    grid = score0.numel()
    new_floats, new_bools = _saved_floats(run(gr.clamped_routing_cost), grid)
    old_floats, _ = _saved_floats(run(_legacy_clamped_routing_cost), grid)
    assert new_floats == 0 and new_bools == 2                # the admissibility mask and the cap mask
    assert old_floats >= 3                                   # the float weights and both clamp inputs
    x = score0.clone().requires_grad_()
    assert _saved_floats(lambda: fam_base.safe_kl_clamp(x, kl_max=1.5), grid) == (0, 1)
    assert _saved_floats(lambda: _legacy_safe_kl_clamp(x, kl_max=1.5), grid)[0] == 2


# ------------------------------------------------------------------------------------------- A138, A155: live route

@pytest.mark.parametrize("reference", REFERENCES)
@pytest.mark.parametrize("kl_max", [16.0, 1.0])
def test_live_route_loss_and_gradients_are_bitwise_the_legacy_chain(reference: str, kl_max: float,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model(routing_kl_max_reference=reference, kl_max=kl_max)
    new = _loss_and_grads(model)
    with monkeypatch.context() as patch:
        _install_legacy_chain(patch)
        legacy = _counting(patch, gr, "clamped_routing_cost")
        old = _loss_and_grads(model)
    assert legacy["n"] == model.cfg.n_layers                                 # the oracle ran in every layer
    _assert_bitwise(new, old)
    if reference == "query_shifted" or kl_max == 16.0:                       # v1 at 1 nat caps every pair
        for name in ("relational_operators.raw_score_q", "relational_operators.routing_key_log_noise"):
            assert float(new[1][name].abs().max()) > 0.0, name               # the router gradient is not trivial
    if kl_max == 1.0:                                                        # the cap binds on the live reference
        snapshot = model.build_diagnostic_snapshot(TOKENS[:1])
        capture = snapshot.relational_capture
        assert bool((capture["score_energy_pre"] >= kl_max).any())


def test_prior_finite_mask_is_computed_once_per_prior_tensor(monkeypatch: pytest.MonkeyPatch) -> None:
    prior = attention_log_prior("causal_alibi", 5, 5, n_heads=2, alibi_slope=1.0)
    shapes: List[Tuple[int, ...]] = []
    real = gr._finite
    monkeypatch.setattr(gr, "_finite", lambda x: (shapes.append(tuple(x.shape)), real(x))[1])
    first, second = gr._finite_prior(prior), gr._finite_prior(prior)
    assert first is second and torch.equal(first, torch.isfinite(prior)) and shapes == [(2, 5, 5)]
    prior[0, 0, 1] = 0.0                                                    # in place: the version counter moves
    third = gr._finite_prior(prior)
    assert third is not first and torch.equal(third, torch.isfinite(prior)) and len(shapes) == 2
    with torch.inference_mode():
        frozen = attention_log_prior("causal", 4, 4)
    assert torch.equal(gr._finite_prior(frozen), torch.isfinite(frozen))   # computed afresh, nothing cached
    assert torch.equal(gr._finite_prior(frozen), torch.isfinite(frozen)) and len(shapes) == 4


def test_model_reads_one_prior_mask_for_all_layers_and_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model()
    shapes: List[Tuple[int, ...]] = []
    real = gr._finite
    monkeypatch.setattr(gr, "_finite", lambda x: (shapes.append(tuple(x.shape)), real(x))[1])
    _loss_and_grads(model)
    _loss_and_grads(model)
    assert shapes.count((2, N, N)) == 1                      # the cached (H, N, N) prior, once over 2 layers x 2 steps
    assert shapes.count((B, 2, N, N)) == 4                   # the cost grid of every layer and step


# ------------------------------------------------------------------------------------------- A140: causal tiles

def test_routing_causal_tiles_is_a_validated_opt_in_field() -> None:
    assert VFE3Config.__dataclass_fields__["routing_causal_tiles"].default is False
    with pytest.raises(ValueError, match="routing_causal_tiles"):
        VFE3Config(**_live_values(routing_causal_tiles="yes"))
    live = VFE3Config(**_live_values(routing_causal_tiles=True))
    assert "routing_causal_tiles" not in live.inactive_fields()
    for off_route in (dict(rope_mode="none"), dict(pair_kl_gemm_tol=1e-3),
                      dict(projected_factorization="gram_certified"),
                      dict(attention_mode="projected_gradient", relational_routing="gaussian_kl", rope_mode="none"),
                      dict(predictive_readout="minimal_agent", attention_residual=False, rope_mode="comparison")):
        assert "routing_causal_tiles" in VFE3Config(**_live_values(**off_route)).inactive_fields(), off_route
    payload = serialize_config(live)
    assert config_from_serialized(payload, source="tiles on") == live
    payload.pop("routing_causal_tiles")                                       # a record from before the field
    assert config_from_serialized(payload, source="before A140").routing_causal_tiles is False
    model = VFEModel(live)
    assert all(ops.routing_causal_tiles is True for ops in model.relational_operators_by_layer)
    with pytest.raises(ValueError, match="routing_causal_tiles"):
        _operators(routing_causal_tiles=1)


def test_causal_tile_plan_covers_the_rows_and_is_gated(_diagonal_route: None) -> None:
    ops = _operators(routing_causal_tiles=True)
    mu = torch.zeros(B, 7, 8, dtype=F64)
    prior = attention_log_prior("causal_alibi", 7, 7, n_heads=2, alibi_slope=1.0, dtype=F64)
    tiles = gr.routing_causal_tile_rows(ops, mu, prior)
    assert tiles == [(0, 2), (2, 4), (4, 6), (6, 7)]                       # ceil(N/4) rows, a ragged last tile
    assert gr.routing_causal_tile_rows(ops, mu, torch.zeros(2, 7, 7, dtype=F64)) is None   # not strictly causal
    assert gr.routing_causal_tile_rows(ops, mu, None) is None
    assert gr.routing_causal_tile_rows(_operators(), mu, prior) is None                     # the toggle off
    assert gr.routing_causal_tile_rows(_operators(routing_causal_tiles=True, pair_kl_gemm_tol=1e-3), mu, prior) is None
    assert gr.routing_causal_tile_rows(_operators(routing_causal_tiles=True, comparison_rope_fraction=.5),
                                       mu, prior) is None                    # P(G) not diagonal: the dense route


def _readout(ops: ro.RelationalOperators, prior: torch.Tensor, n: int = N, *, return_covariance: bool = False,
             kl_max: float = 1.0) -> Tuple[torch.Tensor, ...]:
    r"""The router readout and the gradients of a fixed cotangent with respect to the belief and every parameter."""
    mu0, sigma0, frame = _belief(n, dtype=prior.dtype)
    mu, sigma = mu0.clone().requires_grad_(), sigma0.clone().requires_grad_()
    anchor = {"floor_anchor": sigma0.detach()} if return_covariance else {}
    out = gr.relational_predictive_readout(mu, sigma, frame, ops, tau=1.0, kl_max=kl_max, log_prior=prior,
                                           return_covariance=return_covariance, **anchor)
    outputs = out if return_covariance else (out,)
    g = torch.Generator().manual_seed(9)
    total = sum((y * torch.randn(y.shape, generator=g, dtype=y.dtype)).sum() for y in outputs)
    params = [p for p in ops.parameters() if p.requires_grad]
    grads = torch.autograd.grad(total, [mu, sigma, *params], allow_unused=True)
    return tuple(y.detach() for y in outputs) + tuple(grads)


@pytest.mark.parametrize("dtype, rtol, atol", [(F64, 1e-11, 1e-13), (torch.float32, 2e-5, 1e-5)])
@pytest.mark.parametrize("n, budget", [(6, None), (7, None), (8, 1)])
@pytest.mark.parametrize("reference", ["query_shifted", "row_min"])
def test_causal_tiles_agree_with_the_full_grid(n: int, budget: Optional[int], reference: str, dtype: torch.dtype,
                                               rtol: float, atol: float, _diagonal_route: None,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    if budget is not None:                                                  # one query row per tile, T = N
        monkeypatch.setattr(ro, "_SPREAD_TILE_BYTES", budget)
    prior = attention_log_prior("causal_alibi", n, n, n_heads=2, alibi_slope=1.0, dtype=dtype)
    full = _readout(_operators(dtype, kl_max_reference=reference), prior, n)
    tiled_ops = _operators(dtype, kl_max_reference=reference, routing_causal_tiles=True)
    calls = _counting(monkeypatch, gr, "_causal_routing_rows")
    tiled = _readout(tiled_ops, prior, n)
    assert calls["n"] == 1
    for a, b in zip(tiled, full):
        assert (a is None) == (b is None)
        if a is not None:
            _assert_close_scaled(a, b, rtol, atol)


@pytest.mark.parametrize("n, budget, tiles", [(8, None, 4), (8, 1, 8), (6, None, 3)])
def test_causal_tiles_form_the_causal_fraction_of_the_grid(n: int, budget: Optional[int], tiles: int,
                                                          _diagonal_route: None,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    if budget is not None:
        monkeypatch.setattr(ro, "_SPREAD_TILE_BYTES", budget)
    formed: List[int] = []
    real = gr.routing_cost

    def counting_cost(name: str) -> Callable[..., torch.Tensor]:
        cost = real(name)
        return lambda *args, **kwargs: (lambda D: (formed.append(D.numel()), D)[1])(cost(*args, **kwargs))

    monkeypatch.setattr(gr, "routing_cost", counting_cost)
    prior = attention_log_prior("causal_alibi", n, n, n_heads=2, alibi_slope=1.0, dtype=F64)
    _readout(_operators(routing_causal_tiles=True), prior, n)
    assert len(formed) == tiles
    assert sum(formed) / (B * 2 * n * n) == pytest.approx((tiles + 1) / (2 * tiles))


def test_causal_tiles_keep_the_full_grid_bitwise_off_and_under_a_noncausal_prior(_diagonal_route: None) -> None:
    causal = attention_log_prior("causal_alibi", N, N, n_heads=2, alibi_slope=1.0, dtype=F64)
    open_prior = torch.zeros(2, N, N, dtype=F64)                            # no causal mask: the full grid
    for prior in (causal, open_prior):
        reference = _readout(_operators(), prior)
        assert all(torch.equal(a, b) for a, b in zip(_readout(_operators(), prior), reference))
    off, on = _readout(_operators(), open_prior), _readout(_operators(routing_causal_tiles=True), open_prior)
    assert all(torch.equal(a, b) for a, b in zip(on, off))


def test_causal_tiles_feed_the_moment_matched_covariance(_diagonal_route: None) -> None:
    prior = attention_log_prior("causal", N, N, dtype=F64)
    kwargs = dict(predictive_covariance="moment_matched")
    full = _readout(_operators(**kwargs), prior, return_covariance=True)
    tiled = _readout(_operators(routing_causal_tiles=True, **kwargs), prior, return_covariance=True)
    for a, b in zip(tiled, full):
        assert (a is None) == (b is None)
        if a is not None:
            _assert_close_scaled(a, b, 1e-11, 1e-13)


def _mixed_head_belief(n: int = N) -> Tuple[torch.Tensor, ...]:
    r"""Means and variances (B, n, 6) and a dense block-diagonal frame (B, n, 6, 6) for heads of sizes 2 and 4."""
    g = torch.Generator().manual_seed(21)
    blocks = [torch.linalg.matrix_exp(.3 * torch.randn(B, n, d, d, generator=g, dtype=F64)) for d in (2, 4)]
    U = torch.zeros(B, n, 6, 6, dtype=F64)
    U[..., :2, :2], U[..., 2:, 2:] = blocks
    mu = torch.randn(B, n, 6, generator=g, dtype=F64)
    sigma = .5 + torch.rand(B, n, 6, generator=g, dtype=F64)
    return mu, sigma, CanonicalFrameContext(U, torch.linalg.inv(U), False)


@pytest.mark.parametrize("case", ["mixed_heads", "edge_links"])
def test_causal_tiles_agree_on_mixed_heads_and_edge_links(case: str, _diagonal_route: None,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    prior = attention_log_prior("causal", N, N, dtype=F64)
    results = []
    for tiles in (False, True):
        if case == "mixed_heads":
            torch.manual_seed(3)
            ops = make_relational_operators([2, 4], 5, readout_mode="predictive_content", attention_residual=True,
                                            predictive_covariance="passthrough", routing="noisy_key",
                                            content_link_chart="exponential", kl_max_reference="query_shifted",
                                            rope_comparison_covariance="rotation_invariant",
                                            comparison_rope_base=100.0, routing_causal_tiles=tiles).double()
            mu0, sigma0, frame = _mixed_head_belief()
        else:
            ops = _operators(content_link_edges="lowrank", content_link_edge_rank=2, routing_causal_tiles=tiles)
            with torch.no_grad():                                           # B = 0 would leave the edge links inert
                ops.edge_sender_map.copy_(.1 * torch.randn(ops.edge_sender_map.shape,
                                                           generator=torch.Generator().manual_seed(4), dtype=F64))
            mu0, sigma0, frame = _belief()
        calls = _counting(monkeypatch, gr, "_causal_routing_rows")
        mu, sigma = mu0.clone().requires_grad_(), sigma0.clone().requires_grad_()
        out = gr.relational_predictive_readout(mu, sigma, frame, ops, tau=1.0, kl_max=1.0, log_prior=prior)
        assert calls["n"] == int(tiles)
        weight = torch.randn(out.shape, generator=torch.Generator().manual_seed(8), dtype=F64)
        params = [p for p in ops.parameters() if p.requires_grad]
        results.append((out.detach(), *torch.autograd.grad((out * weight).sum(), [mu, sigma, *params],
                                                           allow_unused=True)))
    for a, b in zip(results[1], results[0]):
        assert (a is None) == (b is None)
        if a is not None:
            _assert_close_scaled(a, b, 1e-11, 1e-13)


def test_predictive_content_of_causal_row_tiles_is_the_full_contraction() -> None:
    ops = _operators()
    mu, _, frame = _belief(7)
    prior = attention_log_prior("causal", 7, 7, dtype=F64)
    beta = torch.softmax(torch.randn(B, 2, 7, 7, generator=torch.Generator().manual_seed(1), dtype=F64) + prior, -1)
    rows = [beta[..., i0:i1, :i1] for i0, i1 in ((0, 2), (2, 4), (4, 6), (6, 7))]
    torch.testing.assert_close(ops.predictive_content(mu, rows, frame), ops.predictive_content(mu, beta, frame),
                               rtol=1e-13, atol=1e-14)
    edged = _operators(content_link_edges="lowrank", content_link_edge_rank=2, rope_comparison_covariance="full",
                       comparison_rope_base=None)
    with pytest.raises(ValueError, match="content_link_edges"):
        edged.predictive_content(mu, rows, frame)


@pytest.mark.parametrize("dtype, rtol, atol", [(F64, 1e-10, 1e-12), (torch.float32, 1e-4, 1e-6)])
def test_live_route_with_causal_tiles_agrees_with_the_full_grid(dtype: torch.dtype, rtol: float, atol: float,
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    full = _model(dtype)
    tiled = _model(dtype, routing_causal_tiles=True)
    tiled.load_state_dict(full.state_dict())
    calls = _counting(monkeypatch, gr, "_causal_routing_rows")
    new = _loss_and_grads(tiled)
    assert calls["n"] == 2                                                   # every layer took the tiles
    old = _loss_and_grads(full)
    assert calls["n"] == 2
    torch.testing.assert_close(new[0], old[0], rtol=rtol, atol=atol)
    assert new[1].keys() == old[1].keys()
    for name in old[1]:
        _assert_close_scaled(new[1][name], old[1][name], rtol, atol, msg=name)


# ------------------------------------------------------------------------------------------- A239, A255: noisy key

def _shifted_and_beta(ops: ro.RelationalOperators, kl_max: float) -> Tuple[torch.Tensor, torch.Tensor]:
    mu, sigma, frame = _belief(seed=12)
    prior = attention_log_prior("causal_alibi", N, N, n_heads=2, alibi_slope=1.0, dtype=F64)
    aux: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        raw = gr.relational_score(mu, sigma, mu, sigma, frame, ops, routing_aux=aux)
        shifted = gr.clamped_routing_cost(raw, ops, kl_max=kl_max, log_prior=prior,
                                          key_log_volume=aux["key_log_volume"])
        beta = attention_weights(shifted, tau=1.0, log_prior=prior)
    return shifted.masked_fill(~torch.isfinite(prior), 0.0), beta


@pytest.mark.parametrize("kl_max", [math.inf, 2.0])
def test_noisy_key_scale_symmetry_scales_the_key_noise_with_the_maps(kl_max: float, _diagonal_route: None) -> None:
    ops = _operators()
    base_shifted, base_beta = _shifted_and_beta(ops, kl_max)
    c = 3.0
    with torch.no_grad():
        ops.raw_score_q.mul_(c)
        ops.raw_score_k.mul_(c)
    maps_shifted, maps_beta = _shifted_and_beta(ops, kl_max)
    with torch.no_grad():
        ops.routing_key_log_noise.add_(2.0 * math.log(c))                  # n_h -> c^2 n_h
    joint_shifted, joint_beta = _shifted_and_beta(ops, kl_max)
    assert float((maps_beta - base_beta).abs().max()) > 1e-3               # the maps alone are no symmetry
    torch.testing.assert_close(joint_beta, base_beta, rtol=0, atol=1e-12)
    torch.testing.assert_close(joint_shifted, base_shifted, rtol=0, atol=1e-11)


def test_noisy_key_capture_records_the_broadened_key_term_medians(_diagonal_route: None) -> None:
    ops = _operators()
    mu, sigma, frame = _belief(seed=13)
    prior = attention_log_prior("causal_alibi", N, N, n_heads=2, alibi_slope=1.0, dtype=F64)
    capture: Dict[str, Any] = {}
    with torch.no_grad():
        gr.relational_predictive_readout(mu, sigma, frame, ops, tau=1.0, kl_max=4.0, log_prior=prior,
                                         capture=capture)
    medians = capture["score_term_medians_pre"]                                # (4, H + 1)
    # The diagonal route captures diag P(G) as (B, N, H, r), a dense route the (B, N, H, r, r) matrices (A142).
    (G_q, G_k, diagonal), = gr.projected_covariance_runs(capture)               # one run of equal heads
    as_diagonal = (lambda G: G) if diagonal else (lambda G: G.diagonal(dim1=-2, dim2=-1))
    g_q = as_diagonal(G_q).double()                                                       # (B, N, H, r) P(G_i)
    g_k = (as_diagonal(G_k).double()
           + ops.routing_key_log_noise.detach().exp().double()[:, None])                 # (B, N, H, r) g_j + n_h
    r = g_q.shape[-1]
    query = (-.5 * (r + g_q.log().sum(-1))).movedim(-1, -2)                               # (B, H, N)
    key = (.5 * g_k.log().sum(-1)).movedim(-1, -2)                                        # (B, H, N)
    trace = .5 * torch.einsum("bihk,bjhk->bhij", g_q, g_k.reciprocal())                 # (B, H, N, N)
    support = torch.isfinite(prior).expand(B, 2, N, N)
    for row, term in enumerate((query.unsqueeze(-1).expand(B, 2, N, N), key.unsqueeze(-2).expand(B, 2, N, N),
                                trace)):
        values = term.masked_fill(~support, float("nan")).movedim(1, 0).reshape(2, -1)
        expected = torch.cat((values.nanmedian(-1).values, values.reshape(1, -1).nanmedian(-1).values))
        torch.testing.assert_close(medians[row], expected, rtol=1e-12, atol=1e-12)
    capture = {}
    with torch.no_grad():
        gr.relational_predictive_readout(mu, sigma, frame, _operators(routing="noisy_query"), tau=1.0, kl_max=4.0,
                                         log_prior=prior, capture=capture)
    assert "score_term_medians_pre" not in capture                            # not a Gaussian KL


def test_live_route_diagnostics_carry_the_cost_term_split() -> None:
    model = _model().eval()
    diagnostics = model.diagnostics(TOKENS[:1])
    for term in ("query_constant", "key_logdet", "trace", "mahalanobis"):
        assert math.isfinite(diagnostics[f"relational_score_term_{term}_median"]), term


# ------------------------------------------------------------------------------------------- A275: ALiBi keywords

@pytest.mark.parametrize("name", ["causal_alibi", "causal_alibi_noself"])
def test_alibi_builders_refuse_the_misspelled_slope(name: str) -> None:
    with pytest.raises(TypeError, match="alibi_slope"):
        attention_log_prior(name, 4, 4, n_heads=2, slope=2.0)
    threaded = attention_log_prior(name, 4, 4, n_heads=2, alibi_slope=2.0, window=3)   # the model's keywords
    assert torch.equal(threaded, attention_log_prior(name, 4, 4, n_heads=2, alibi_slope=2.0))
    assert not torch.equal(threaded, attention_log_prior(name, 4, 4, n_heads=2))


# ------------------------------------------------------------------------------------------- A193: full-link episode

def _episode(update: str, *, full: bool = False, n_iter: int = 1, record: bool = False,
             capture: bool = False) -> Tuple[Any, ...]:
    r"""One full-link receiver episode on the float64 fixture of tests/test_full_link_gradients.py."""
    mu, sigma, mu_p, sigma_p, frame, ops, prior = _fixture(full=full)
    ops.readout_mode = "full_link_update"
    ops.receiver_update = update
    belief = BeliefState(mu, sigma, torch.zeros(1, 3, 2, dtype=mu.dtype))
    grad_record: Optional[dict] = {} if record else None
    out = e_step_module.e_step(belief, mu_p, sigma_p, None, log_prior=prior, relational_operators=ops,
                               relational_frame=frame, relational_capture={} if capture else None,
                               grad_record=grad_record, family="gaussian_full" if full else "gaussian_diagonal",
                               e_q_mu_lr=.05, e_q_sigma_lr=.05, e_mu_q_trust=None, e_sigma_q_trust=50.0,
                               sigma_max=None, n_iter=n_iter, lambda_beta=1.0)
    return out, grad_record, (mu, sigma, mu_p, sigma_p, frame, ops, prior)


@pytest.mark.parametrize("update", ["closed_form", "closed_form_profiled", "euler"])
@pytest.mark.parametrize("record", [False, True])
def test_full_link_episode_forms_the_force_only_where_it_is_read(update: str, record: bool,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _counting(monkeypatch, e_step_module, "full_link_gradients")
    _, grad_record, _ = _episode(update, n_iter=2, record=record)
    assert calls["n"] == (2 if update == "euler" or record else 0)
    if record:
        assert grad_record["mu"] is not None and float(grad_record["mu"]) > 0.0


@pytest.mark.parametrize("n_iter", [1, 2, 3])
def test_full_link_first_iteration_reuses_the_built_context(n_iter: int, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _counting(monkeypatch, e_step_module, "full_link_context_at")
    _episode("closed_form", n_iter=n_iter)
    assert calls["n"] == n_iter - 1


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("update", ["closed_form", "closed_form_profiled", "euler"])
def test_full_link_episode_is_bitwise_the_recomputed_iterate(full: bool, update: str) -> None:
    out, _, (mu, sigma, mu_p, sigma_p, frame, ops, prior) = _episode(update, full=full)
    context = full_link.build_full_link_context(mu, sigma, mu_p, sigma_p, frame, ops, tau=1.0, lambda_beta=1.0,
                                                receiver_noise=ops.receiver_noise, kl_max=100.0, log_prior=prior)
    iterate = full_link.full_link_context_at(context, mu, sigma)            # the 5bb0a57 first iteration
    if update == "euler":
        from vfe3.families.base import get_family
        from vfe3.geometry.retraction import retract_spd_affine
        grad_mu, grad_sigma = full_link.full_link_gradients(mu, sigma, iterate)
        nat_mu, nat_sigma = get_family("gaussian_full" if full else "gaussian_diagonal")(mu, sigma).natural_gradient(
            grad_mu, grad_sigma, eps=1e-6)
        ref_mu = mu - .05 * nat_mu
        ref_sigma = retract_spd_affine(sigma, -.05 * nat_sigma, mu.dim(), trust_region=50.0, eps=1e-6, sigma_max=None)
    elif update == "closed_form":
        ref_mu, ref_sigma = full_link.full_link_minimizer(mu, sigma, iterate, damping=ops.receiver_damping,
                                                          update_sigma=True, eps=1e-6, sigma_max=None)
    else:
        ref_mu, ref_sigma = full_link.full_link_profiled_minimizer(mu, sigma, iterate, damping=ops.receiver_damping,
                                                                   update_sigma=True, eps=1e-6, sigma_max=None)
    assert torch.equal(out.mu, ref_mu) and torch.equal(out.sigma, ref_sigma)
    leaves = [mu, sigma, *(p for p in ops.parameters() if p.requires_grad)]
    new = torch.autograd.grad(out.mu.sum() + out.sigma.square().sum(), leaves, allow_unused=True)
    old = torch.autograd.grad(ref_mu.sum() + ref_sigma.square().sum(), leaves, allow_unused=True)
    for a, b in zip(new, old):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-14)


def test_full_prior_covariance_is_factored_once(monkeypatch: pytest.MonkeyPatch) -> None:
    mu, sigma, mu_p, sigma_p0, frame, ops, prior = _fixture(full=True)
    real = torch.linalg.cholesky

    def run(sigma_p: torch.Tensor) -> Tuple[Any, int]:
        seen = {"n": 0}

        def spy(A: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            seen["n"] += int(A.shape == sigma_p.shape and torch.equal(A.detach(), sigma_p.detach()))
            return real(A, *args, **kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(torch.linalg, "cholesky", spy)
            context = full_link.build_full_link_context(mu, sigma, mu_p, sigma_p, frame, ops, receiver_noise=.7,
                                                        log_prior=prior)
        return context, seen["n"]

    sigma_p = sigma_p0.clone().requires_grad_()
    context, factorizations = run(sigma_p)
    assert factorizations == 1
    factor = real(sigma_p0)
    profile = context.profile
    assert torch.equal(profile.prior_precision, torch.cholesky_inverse(factor))
    assert torch.equal(profile.prior_logdet, 2 * factor.diagonal(dim1=-2, dim2=-1).log().sum(-1))
    # The sigma_p gradient through both terms agrees with the two-factor form to rounding.
    weight = torch.randn(profile.prior_precision.shape, generator=torch.Generator().manual_seed(3), dtype=F64)
    (got,) = torch.autograd.grad((profile.prior_precision * weight).sum() + profile.prior_logdet.sum(), sigma_p)
    reference = sigma_p0.clone().requires_grad_()
    two = (torch.cholesky_inverse(real(reference)) * weight).sum() + (
        2 * real(reference).diagonal(dim1=-2, dim2=-1).log().sum(-1)).sum()
    (want,) = torch.autograd.grad(two, reference)
    torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-13)


# ------------------------------------------------------------------------------------------- A334: retry count

def _projected_gradient_model() -> VFEModel:
    torch.manual_seed(0)
    cfg = test_config(vocab_size=7, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=1, n_e_steps=1,
                      family="gaussian_diagonal", gauge_group="block_glk", skip_belief_sigma_update=True,
                      warmup_steps=0, max_steps=2, log_interval=0, eval_interval=0, pos_phi="learned",
                      use_block_mlp=True, block_mlp_expansion=1, attention_mode="projected_gradient",
                      attention_residual=False, predictive_covariance="passthrough")
    return VFEModel(cfg).eval()


def test_receiving_law_retry_count_reaches_the_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _projected_gradient_model()
    ids = torch.tensor([[0, 1, 2, 3]])
    healthy = model.diagnostics(ids)
    assert healthy["relational_receiver_law_retry_count"] == 0.0
    real_spd, real_cholesky = full_link._spd_precision, torch.linalg.cholesky_ex
    forced = {"n": 0}

    def failing_once(A: torch.Tensor, *args: Any, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        L, info = real_cholesky(A, *args, **kwargs)
        if A.dtype == torch.float32 and forced["n"] == 0:
            forced["n"] += 1
            info = torch.ones_like(info)                    # every entry of this float32 batch reads as failed
        return L, info

    def spd_with_forced_retry(target: torch.Tensor) -> torch.Tensor:
        with monkeypatch.context() as patch:
            patch.setattr(torch.linalg, "cholesky_ex", failing_once)
            return real_spd(target)

    monkeypatch.setattr(full_link, "_spd_precision", spd_with_forced_retry)
    before = int(full_link.receiver_law_retry_counter(torch.device("cpu")))
    snapshot = model.build_diagnostic_snapshot(ids)
    retried = int(full_link.receiver_law_retry_counter(torch.device("cpu"))) - before
    assert forced["n"] == 1 and retried > 0
    assert int(snapshot.relational_capture["relational_receiver_law_retry_count"]) == retried
    diagnostics = model.diagnostics(ids, snapshot=snapshot)
    assert diagnostics["relational_receiver_law_retry_count"] == float(retried)
    assert "val_relational_receiver_law_retry_count" in T._VAL_RELATIONAL_DIAG_KEYS
    assert _model().eval().diagnostics(TOKENS[:1])["relational_receiver_law_retry_count"] == 0.0   # predictive route


# ------------------------------------------------------------------------------------------- A344: _transport

def test_transport_materializes_without_right_phi_whatever_materialize_is() -> None:
    model = _projected_gradient_model()
    group = model.group
    g = torch.Generator().manual_seed(2)
    phi = 0.2 * torch.randn(1, 4, group.generator_shape[0], generator=g)
    dense = e_step_module._transport(phi, group, materialize=True)
    lazy = e_step_module._transport(phi, group, materialize=False)
    assert torch.is_tensor(lazy) and lazy.shape == (1, 4, 4, 4, 4) and torch.equal(lazy, dense)
    right = 0.1 * torch.randn(1, 4, group.generator_shape[0], generator=g)
    factored = e_step_module._transport(phi, group, right_phi=right, materialize=False)
    assert isinstance(factored, FactoredTransport)
    assert torch.equal(factored.to_dense_omega(), e_step_module._transport(phi, group, right_phi=right))
