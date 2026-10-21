r"""Audit 2026-10-20 lane L1 (router): items 1, 9, 38 and 39.

Item 1: ``routing_kl_max_reference='query_shifted'`` clamps :math:`D_{ij} - c_i - m_i`, with :math:`m_i` the admissible
row mean of the key log-volumes, which is invariant under a joint rescaling of a head's comparison maps; the 2026-10-18
form, which kept the key log-determinant, is retained as ``'query_shifted_v1'`` and saved configs reload to it. Item 9:
``relational_score_weight_decay`` decays the comparison maps only. Item 38: per-head router cost-term medians reach
``diagnostics/score_terms.csv``. Item 39: ``relational_routing_free_energy`` is the free energy of the absolute cost
under the shifted references. Tiny CPU models only (K = 8, two heads, N = 6, vocab 9), float64 where invariance is
asserted.
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
from vfe3.config import config_from_serialized, historical_default_pins, serialize_config
from vfe3.gradients.relational import (
    _key_log_volume, clamped_routing_cost, query_shifted_row_reference,
)
from vfe3.model.model import VFEModel
from vfe3.posthoc.klcap import causal_support

F64 = torch.float64
TOKENS = torch.tensor([[0, 1, 2, 3, 4, 5], [5, 6, 7, 8, 0, 1]])
TARGETS = torch.tensor([[1, 2, 3, 4, 5, 6], [6, 7, 8, 0, 1, 2]])
R = 4                                                                        # comparison rank d = K / H = 8 / 2


def _model(reference: str, rope_mode: str, kl_max: float, **overrides: Any) -> VFEModel:
    r"""A one-layer kernel-router model in float64 (the same initialization for every reference and cap)."""
    values: Dict[str, Any] = dict(vocab_size=9, embed_dim=8, n_heads=2, max_seq_len=6, n_layers=1, phi_scale=0.4,
                                  rope_mode=rope_mode, routing_kl_max_reference=reference, kl_max=kl_max,
                                  predictive_readout="kernel")
    values.update(overrides)
    torch.manual_seed(3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VFEModel(tiny_config(**values)).double()


def _scale_maps(model: VFEModel, c: float) -> None:
    r"""The joint rescaling :math:`(S_Q, S_K) \to c(S_Q, S_K)` of every head's comparison maps."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "raw_score_q" in name or "raw_score_k" in name:
                p.mul_(c)


def _router_capture(model: VFEModel) -> Dict[str, torch.Tensor]:
    snapshot = model.build_diagnostic_snapshot(TOKENS)
    cap = model._layer_capture(snapshot, 0)
    support = causal_support(cap["beta_pre"], snapshot.log_prior)               # (B, H, N, N)
    return {"score": cap["score_energy_pre"].double(), "beta": cap["beta_pre"].double(), "support": support}


def _router_grads(model: VFEModel) -> Dict[str, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    _, loss, _ = model(TOKENS, TARGETS)
    loss.backward()
    return {name: p.grad.detach().clone() for name, p in model.named_parameters() if "raw_score" in name}


def _binding_cap(reference: str, rope_mode: str) -> float:
    r"""A cap midway between the two middle admissible clamped costs of the unscaled model, so it binds on about half
    and no entry sits at the kink of the clamp (where rounding under the rescaling could flip it)."""
    values = _router_capture(_model(reference, rope_mode, 1e6))
    rows = values["support"].sum(-1, keepdim=True) >= 2
    costs = values["score"][values["support"] & rows].sort().values
    middle = costs.numel() // 2
    assert float(costs[middle] - costs[middle - 1]) > 1e-6
    return float(0.5 * (costs[middle - 1] + costs[middle]))


# ------------------------------------------------------------------------------------------------ item 1


@pytest.mark.parametrize("rope_mode", ["comparison_invariant", "none"], ids=["diagonal_route", "dense_route"])
def test_query_shifted_clamped_cost_and_router_gradient_are_invariant_under_joint_rescaling(rope_mode: str) -> None:
    kl_max = _binding_cap("query_shifted", rope_mode)
    base = _model("query_shifted", rope_mode, kl_max)
    reference = _router_capture(base)
    capped = (reference["score"] >= kl_max) & reference["support"]
    assert 0 < int(capped.sum()) < int(reference["support"].sum())              # the cap binds on part of the support
    grads = _router_grads(base)
    assert grads and all(float(g.abs().max()) > 0 for g in grads.values())
    for c in (0.3, 4.0):
        scaled = _model("query_shifted", rope_mode, kl_max)
        _scale_maps(scaled, c)
        values = _router_capture(scaled)
        torch.testing.assert_close(values["score"], reference["score"], rtol=0, atol=1e-10)
        torch.testing.assert_close(values["beta"], reference["beta"], rtol=0, atol=1e-12)
        # L(cS) = L(S), so grad L at cS is grad L at S divided by c
        scaled_grads = _router_grads(scaled)
        for name, grad in grads.items():
            torch.testing.assert_close(c * scaled_grads[name], grad, rtol=1e-8, atol=1e-12)


@pytest.mark.parametrize("rope_mode", ["comparison_invariant", "none"], ids=["diagonal_route", "dense_route"])
def test_query_shifted_v1_moves_by_r_ln_c_under_joint_rescaling(rope_mode: str) -> None:
    kl_max = 1e6                                                                # the tested cost itself, uncapped
    reference = _router_capture(_model("query_shifted_v1", rope_mode, kl_max))
    c = 2.0
    scaled = _model("query_shifted_v1", rope_mode, kl_max)
    _scale_maps(scaled, c)
    values = _router_capture(scaled)
    below = (values["score"] < kl_max) & (reference["score"] < kl_max) & reference["support"]
    assert bool(below.any())
    shift = (values["score"] - reference["score"])[below]
    torch.testing.assert_close(shift, torch.full_like(shift, R * math.log(c)), rtol=0, atol=1e-9)


@pytest.mark.parametrize("rope_mode", ["comparison_invariant", "none"], ids=["diagonal_route", "dense_route"])
def test_a_head_pushed_over_the_cap_by_scale_alone_is_no_longer_fully_clamped(rope_mode: str) -> None:
    r"""Under ``query_shifted_v1`` a joint rescaling alone pushes every admissible entry over the cap, the head routes by
    the prior with zero router gradient (the reference's dead heads); under ``query_shifted`` the same weights keep
    every row below the cap and the router gradient nonzero."""
    unscaled = _router_capture(_model("query_shifted_v1", rope_mode, 1e6))
    support = unscaled["support"]
    high = float(unscaled["score"][support].max())
    low = float(unscaled["score"][support].min())
    kl_max = high + 1.0                                                         # nothing capped at c = 1
    c = math.exp((kl_max - low + 2.0) / R)                                      # every entry moves over the cap
    for reference, dead in (("query_shifted_v1", True), ("query_shifted", False)):
        model = _model(reference, rope_mode, kl_max)
        _scale_maps(model, c)
        values = _router_capture(model)
        capped = (values["score"] >= kl_max) & support
        rows = support.sum(-1) >= 2
        fully = rows & (capped.sum(-1) == support.sum(-1))
        grads = _router_grads(model)
        if dead:
            assert bool(fully[rows].all())
            assert all(float(g.abs().max()) == 0.0 for g in grads.values())
        else:
            assert not bool(fully.any()) and not bool(capped.any())
            assert all(float(g.abs().max()) > 0.0 for g in grads.values())


def test_query_shifted_beta_is_the_softmax_and_matches_v1_below_the_cap() -> None:
    absolute = _model("absolute", "comparison_invariant", 1e6)
    absolute_beta = _router_capture(absolute)["beta"]
    for reference in ("query_shifted", "query_shifted_v1"):
        torch.testing.assert_close(_router_capture(_model(reference, "comparison_invariant", 1e6))["beta"],
                                   absolute_beta, rtol=0, atol=1e-12)


def test_row_reference_is_the_admissible_mean_of_finite_key_volumes() -> None:
    score = torch.zeros(1, 1, 3, 3, dtype=F64)
    score[..., 2, 1] = float("nan")                                            # a failed pair leaves its row mean
    log_prior = torch.zeros(3, 3, dtype=F64).masked_fill(torch.ones(3, 3, dtype=torch.bool).triu(1), -math.inf)
    volume = torch.tensor([[[1.0, 3.0, float("nan")]]], dtype=F64)             # (B, H, N); key 2 is not PD
    m = query_shifted_row_reference(score, volume, log_prior).reshape(-1)
    assert m.tolist() == [1.0, 2.0, 1.0]


def test_key_log_volume_on_both_routes_and_under_noisy_key() -> None:
    g = torch.Generator().manual_seed(5)
    var = 0.5 + torch.rand(2, 3, 2, R, generator=g, dtype=F64)               # (B, N, H, r)
    expected = .5 * var.log().sum(-1).movedim(-1, -2)                           # (B, H, N)
    torch.testing.assert_close(_key_log_volume(var, "gaussian_kl", diagonal=True), expected, rtol=0, atol=1e-14)
    torch.testing.assert_close(_key_log_volume(torch.diag_embed(var), "gaussian_kl", diagonal=False), expected,
                               rtol=0, atol=1e-13)
    noise = torch.tensor([0.5, 2.0], dtype=F64)
    broadened = .5 * (var + noise[:, None]).log().sum(-1).movedim(-1, -2)
    torch.testing.assert_close(_key_log_volume(torch.diag_embed(var), "noisy_key", diagonal=False, key_noise=noise),
                               broadened, rtol=0, atol=1e-13)
    assert float(_key_log_volume(var, "mean_distance", diagonal=True).abs().max()) == 0.0


def test_query_shifted_clamp_requires_the_key_volumes() -> None:
    with pytest.raises(ValueError, match="key_log_volume"):
        clamped_routing_cost(torch.zeros(1, 2, 2), SimpleNamespace(kl_max_reference="query_shifted"), kl_max=1.0)


def test_saved_query_shifted_configs_reload_as_v1_and_new_ones_keep_query_shifted() -> None:
    payload = serialize_config(tiny_config(routing_kl_max_reference="query_shifted"))
    assert payload["routing_kl_max_reference"] == "query_shifted" and "relational_score_weight_decay" in payload
    assert config_from_serialized(payload, source="new").routing_kl_max_reference == "query_shifted"
    old = {k: v for k, v in payload.items() if k != "relational_score_weight_decay"}   # saved before 2026-10-21
    assert historical_default_pins(old)["routing_kl_max_reference"] == "query_shifted_v1"
    reloaded = config_from_serialized(old, source="old")
    assert reloaded.routing_kl_max_reference == "query_shifted_v1"
    assert reloaded.relational_score_weight_decay == 0.0
    for reference in ("absolute", "row_min"):
        older = {**old, "routing_kl_max_reference": reference}
        assert config_from_serialized(older, source="old").routing_kl_max_reference == reference
    v1 = serialize_config(tiny_config(routing_kl_max_reference="query_shifted_v1"))
    assert config_from_serialized(v1, source="v1").routing_kl_max_reference == "query_shifted_v1"


def test_v1_route_is_the_pre_fix_clamp_bitwise() -> None:
    r"""``query_shifted_v1`` is the 2026-10-18 clamp, :math:`\min(D_{ij} - c_i, \kappa)`, on the model's own raw cost."""
    kl_max = _binding_cap("query_shifted_v1", "comparison_invariant")
    model = _model("query_shifted_v1", "comparison_invariant", kl_max)
    snapshot = model.build_diagnostic_snapshot(TOKENS)
    cap = model._layer_capture(snapshot, 0)
    # the raw row minimum is that of D - c_i; the clamped minimum is its clamp from above
    support = causal_support(cap["beta_pre"], snapshot.log_prior)
    rows = support.sum(-1) >= 2
    clamped_min = cap["score_energy_pre"].double().masked_fill(~support, math.inf).amin(-1)
    torch.testing.assert_close(clamped_min[rows], cap["score_raw_row_min_pre"][rows].clamp(max=kl_max),
                               rtol=0, atol=1e-12)


# ------------------------------------------------------------------------------------------------ item 9


def test_score_weight_decay_off_keeps_the_single_undecayed_group() -> None:
    model = _model("absolute", "none", 1e6, learn_attention_temperature=True)
    groups = [g for g in T.build_optimizer(model, model.cfg).param_groups if g.get("lr_aux_role") == "relational_score"]
    assert len(groups) == 1 and groups[0]["weight_decay"] == 0.0
    names = {id(p): n for n, p in model.named_parameters()}
    assert any("routing_log_temperature" in names[id(p)] for p in groups[0]["params"])


def test_score_weight_decay_reaches_the_comparison_maps_only() -> None:
    model = _model("absolute", "none", 1e6, learn_attention_temperature=True, relational_score_weight_decay=0.5)
    optimizer = T.build_optimizer(model, model.cfg)
    names = {id(p): n for n, p in model.named_parameters()}
    decayed = [g for g in optimizer.param_groups if g.get("weight_decay", 0.0) == 0.5]
    assert len(decayed) == 1 and decayed[0]["lr_aux_role"] == "relational_score"
    assert sorted(names[id(p)].rsplit(".", 1)[-1] for p in decayed[0]["params"]) == ["raw_score_k", "raw_score_q"]
    scalars = [g for g in optimizer.param_groups
               if any("routing_log_temperature" in names[id(p)] for p in g["params"])]
    assert len(scalars) == 1 and scalars[0]["weight_decay"] == 0.0 and "lr_aux_role" not in scalars[0]
    # one step with zero gradients: AdamW's decoupled decay alone scales the maps by 1 - lr * lambda
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    optimizer.step()
    for group in optimizer.param_groups:
        for p in group["params"]:
            name = names[id(p)]
            factor = 1.0 - group["lr"] * group["weight_decay"] if "raw_score" in name else None
            if factor is not None:
                torch.testing.assert_close(p.detach(), before[name] * factor, rtol=1e-12, atol=0)
            elif "routing_log_temperature" in name:
                assert torch.equal(p.detach(), before[name])


def test_score_weight_decay_is_validated() -> None:
    with pytest.raises(ValueError, match="relational_score_weight_decay"):
        tiny_config(relational_score_weight_decay=-0.1)


# ------------------------------------------------------------------------------------------------ item 38


def test_per_head_score_terms_reach_their_csv_rows() -> None:
    model = _model("query_shifted", "comparison_invariant", 1e6).float()
    sums: Dict[Any, List[float]] = {}
    T._residual_and_attention_rows(model, TOKENS, step=7, device=torch.device("cpu"), score_terms_out=sums)
    assert sorted(sums) == [(0, 0), (0, 1)]
    snapshot = model.build_diagnostic_snapshot(TOKENS)
    medians = model._layer_capture(snapshot, 0)["score_term_medians_pre"]      # (4, H + 1)
    rows = T._score_term_rows(sums, 7)
    assert [tuple(row[k] for k in ("step", "layer", "head")) for row in rows] == [(7, 0, 0), (7, 0, 1)]
    assert all(set(row) == set(T.SCORE_TERM_FIELDS) for row in rows)
    for row in rows:
        for index, term in enumerate(("query_constant", "key_logdet", "trace", "mahalanobis")):
            assert row[term] == pytest.approx(float(medians[index, row["head"]]), rel=1e-6, abs=1e-6)
        assert row["n_chunks"] == 1


def test_dense_route_writes_no_score_term_rows() -> None:
    model = _model("absolute", "none", 1e6).float()
    sums: Dict[Any, List[float]] = {}
    T._residual_and_attention_rows(model, TOKENS, step=1, device=torch.device("cpu"), score_terms_out=sums)
    assert sums == {}


# ------------------------------------------------------------------------------------------------ item 39


@pytest.mark.parametrize("rope_mode", ["comparison_invariant", "none"], ids=["diagonal_route", "dense_route"])
def test_routing_free_energy_reads_the_absolute_cost_under_the_shifted_references(rope_mode: str) -> None:
    def free_energy(reference: str) -> float:
        model = _model(reference, rope_mode, 1e6)
        return float(model.diagnostics(TOKENS)["relational_routing_free_energy"])

    absolute = free_energy("absolute")
    assert math.isfinite(absolute)
    for reference in ("query_shifted", "query_shifted_v1"):
        assert free_energy(reference) == pytest.approx(absolute, rel=1e-9, abs=1e-9)


def test_query_constant_capture_is_the_dropped_constant() -> None:
    model = _model("query_shifted", "comparison_invariant", 1e6)
    snapshot = model.build_diagnostic_snapshot(TOKENS)
    cap = model._layer_capture(snapshot, 0)
    G_q = cap["projected_covariance_pre"].diagonal(dim1=-2, dim2=-1).double()     # (B, N, H, r)
    expected = (-.5 * (R + G_q.log().sum(-1))).movedim(-1, -2)                    # (B, H, N)
    torch.testing.assert_close(cap["score_query_constant_pre"], expected, rtol=0, atol=1e-12)
    assert cap["score_row_shift_pre"].shape == expected.shape
