r"""Audit 2026-10-09 round 2, paper items 3, 4/12, 16, 18, 19, 27 and 44 (paper part).

The planned paper base (K = 128, four heads, six layers, ALiBi without RoPE, ``row_min``, no attention
residual, chunked diagonal decoder) is simulated by a tiny base with the same switches (K = 4, two
heads, two layers; six layers for the rank-collapse check). Nothing is trained.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import pytest
import torch

import paper_arms as pa
from tests.test_paper_arms_20260928 import _tiny_cfg, _write_base
from vfe3.paper import tables as pt
from vfe3.paper.runs import Arm, Manifest, RunRecord

POST_NORM = "Per-type read table, ALiBi, shared post-norm"
TOTAL_MATCHED = "Matched total parameters"
RESIDUAL_ADDED = "Attention residual added"
BLOCK_LEARNED = "Head-block family, learned frames"
BLOCK_CHOLESKY = "Head-block family, frame-free Cholesky table"
BLOCK_REFERENCE = "Head-block family, normalization, MLP and decoder in reference coordinates"
PAIRS = [(6, 3), (23, 23), (54, 54)]


def _planned(**over: Any) -> Any:
    r"""Tiny twin of the planned paper base: ALiBi, no RoPE, row_min, no residual, chunked diagonal decode."""
    values = dict(attention_residual=False, beta_attention_prior="causal_alibi", rope_mode="none",
                  routing_kl_max_reference="row_min", decode_mode="diagonal_chunked", sigma_weight_decay=0.0,
                  pos_phi="none", predictive_covariance="moment_matched",
                  content_link_chart="exponential")
    values.update(over)
    return _tiny_cfg(**values)


def _base(tmp_path: Path, name: str = "planned", **over: Any) -> pa.Base:
    return pa.load_base(name, _write_base(tmp_path / name, _planned(**over)))


def _plan(base: pa.Base, label: str, **kwargs: Any) -> pa.ArmPlan:
    return pa.build_plan(base, budget="base_protocol", seeds=[(1, 1)], labels=[label],
                         baseline_overrides={"lr": 3e-3}, **kwargs)[0]


def _own_token_sensitivity(cfg: Any) -> float:
    r"""Frobenius norm of :math:`\partial \ell_{N-1} / \partial E_{x_{N-1}}` at initialization (distinct tokens)."""
    from torch.func import functional_call

    from vfe3.baseline import BaselineTransformer

    torch.manual_seed(0)
    model = BaselineTransformer(cfg)
    model.set_unigram_log_prior(torch.ones(cfg.vocab_size))
    n = cfg.max_seq_len
    ids = torch.arange(n).unsqueeze(0)                                                     # (1, N) distinct
    E = model.tok_embed.detach().clone()                                                   # (V, K)

    def last_logits(table: torch.Tensor) -> torch.Tensor:                                  # (V,)
        return functional_call(model, {"tok_embed": table}, (ids,))[0, -1]

    J = torch.autograd.functional.jacobian(last_logits, E)                                 # (V, V, K)
    return float(J[:, ids[0, -1]].norm())


# ---- item 3: the shared post-norm arm keeps the attention residual ------------------------------

def test_post_norm_arm_keeps_the_residual_and_is_not_rank_collapsed_at_init(tmp_path: Path) -> None:
    import dataclasses

    from vfe3.baseline import BaselineConfig

    import train_baseline  # noqa: F401
    base = _base(tmp_path, n_layers=6)
    plan = _plan(base, POST_NORM, counts=False)
    assert plan.status == "ready" and plan.cfg.norm_placement == "post_shared" and plan.cfg.n_layers == 6
    assert plan.cfg.attention_residual is True and plan.diff["attention_residual"] == (False, True)
    assert any("attention residual kept" in d for d in plan.deviations)
    assert any(note.startswith("declared difference: attention residual kept") for note in plan.notes)
    manifest = pa.build_manifest(tmp_path / "exp", [plan], base=base, budget="base_protocol", seeds=[(1, 1)])
    row = manifest["paper_arms"]["rows"][POST_NORM]
    assert row["changes"]["attention_residual"] == ["False", "True"] and row["declared_deviations"] == plan.deviations

    sensitivity = _own_token_sensitivity(plan.cfg)
    collapsed = _own_token_sensitivity(dataclasses.replace(plan.cfg, attention_residual=False))
    assert sensitivity > 1.0                    # O(1) at six layers with the residual
    assert collapsed < 1e-3 * sensitivity       # the counterfactual LN(Attn(h)) arm starts collapsed
    # The option itself is unchanged: matched_to still copies a missing residual unless overridden.
    assert BaselineConfig.matched_to(base.cfg, norm_placement="post_shared").attention_residual is False


# ---- item 19: matched_to refuses bases whose matching it cannot honor --------------------------

def test_matched_to_refuses_tied_and_block_cholesky_bases_and_notes_learned_temperature(tmp_path: Path) -> None:
    from vfe3.baseline import BaselineConfig

    import train_baseline  # noqa: F401
    tied = _planned(tie_comparison_maps=True)
    with pytest.raises(ValueError, match="tie_comparison_maps"):
        BaselineConfig.matched_to(tied)
    # The tie removes one K x K comparison map per layer.
    untied = pa._vfe_counts(_planned())["context_processing"]
    assert untied - pa._vfe_counts(tied)["context_processing"] == tied.n_layers * tied.embed_dim ** 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cholesky = _planned(family="gaussian_block", prior_covariance_table="block_cholesky", gauge_transport="off")
    with pytest.raises(ValueError, match="prior_covariance_table"):
        BaselineConfig.matched_to(cholesky)
    plan = _plan(_base(tmp_path, learn_attention_temperature=True), "Matched context-processing parameters",
                 counts=False)
    assert plan.status == "ready"
    assert any("H x L = 4" in note for note in plan.notes)
    refused = _plan(_base(tmp_path, "tied", tie_comparison_maps=True), "Matched context-processing parameters",
                    counts=False)
    assert refused.status == "invalid" and "tie_comparison_maps" in refused.reason


# ---- item 18: the total-matched arm transfers the hidden-matrix learning rate -------------------

def test_total_matched_arm_scales_the_hidden_lr_and_records_it_untuned(tmp_path: Path) -> None:
    from vfe3.baseline import BaselineTransformer, build_optimizer

    import train_baseline  # noqa: F401
    base = _base(tmp_path, vocab_size=200)
    plan = _plan(base, TOTAL_MATCHED)
    K0, K = base.cfg.embed_dim, plan.cfg.embed_dim
    assert plan.status == "ready" and K > K0
    assert plan.cfg.hidden_lr_scale == pytest.approx(K0 / K) and plan.diff["hidden_lr_scale"] == (1.0, K0 / K)
    assert any(d.startswith("lr not tuned at this width") for d in plan.deviations)
    manifest = pa.build_manifest(tmp_path / "exp", [plan], base=base, budget="base_protocol", seeds=[(1, 1)])
    assert manifest["paper_arms"]["rows"][TOTAL_MATCHED]["declared_deviations"] == plan.deviations
    assert TOTAL_MATCHED in pa.TIER_A and TOTAL_MATCHED not in pa.TIER_B

    model = BaselineTransformer(plan.cfg)
    groups = build_optimizer(model, plan.cfg).param_groups
    hidden = {id(p) for name, p in model.named_parameters() if name.startswith("blocks.") and p.dim() >= 2}
    scaled = [g for g in groups if g["lr"] == pytest.approx(plan.cfg.lr * K0 / K)]
    assert len(scaled) == 1 and {id(p) for p in scaled[0]["params"]} == hidden
    assert scaled[0]["weight_decay"] == plan.cfg.weight_decay
    for name in ("tok_embed", "decoder.output_proj_weight"):
        parameter = dict(model.named_parameters())[name]
        assert next(g for g in groups if any(p is parameter for p in g["params"]))["lr"] == plan.cfg.lr
    # Scale 1 keeps the historical groups (and so optimizer-state layout).
    plain = _plan(base, "Matched context-processing parameters", counts=False).cfg
    assert plain.hidden_lr_scale == 1.0 and len(build_optimizer(BaselineTransformer(plain), plain).param_groups) == 2


# ---- item 16: a single-change arm adding the attention residual ---------------------------------

def test_residual_added_arm_is_one_change_on_a_no_residual_base(tmp_path: Path) -> None:
    spec = pa.ARMS[RESIDUAL_ADDED]
    assert spec["group"] == pa._RETRAINING and RESIDUAL_ADDED not in (*pa.TIER_A, *pa.TIER_B)
    plan = _plan(_base(tmp_path), RESIDUAL_ADDED, counts=False)
    assert plan.status == "ready" and plan.diff == {"attention_residual": (False, True)}
    with_residual = _plan(_base(tmp_path, "res", attention_residual=True), RESIDUAL_ADDED, counts=False)
    assert with_residual.status == "inapplicable" and "attention_residual" in with_residual.reason


# ---- items 4 / 12: head-block arms that make the frame claim testable --------------------------

def test_head_block_arms_resolve_on_the_planned_alibi_base(tmp_path: Path) -> None:
    base = _base(tmp_path)
    plans = {label: _plan(base, label) for label in (BLOCK_LEARNED, BLOCK_CHOLESKY, BLOCK_REFERENCE)}
    for label, plan in plans.items():
        assert plan.status == "ready", (label, plan.reason)
        assert plan.params is not None and plan.params["total"] > 0          # the arm's model builds
        assert pa.ARMS[label]["group"] == pa._RETRAINING and label not in (*pa.TIER_A, *pa.TIER_B)
    assert plans[BLOCK_LEARNED].diff == {"family": ("gaussian_diagonal", "gaussian_block")}
    assert set(plans[BLOCK_CHOLESKY].diff) - set(pa.GAUGE_OFF_COERCED) == {
        "family", "prior_covariance_table", "gauge_transport"}
    assert set(plans[BLOCK_REFERENCE].diff) == {"family", "block_coordinates", "decode_coordinates"}
    block_base = _base(tmp_path, "block", family="gaussian_block")
    assert _plan(block_base, BLOCK_LEARNED, counts=False).status == "inapplicable"


# ---- item 27: no capacity control, no standing claim ---------------------------------------------

def _arm(label: str, ces: Any, **kwargs: Any) -> Arm:
    from vfe3.paper.runs import RunRecord as Record

    runs = [Record(run_dir=Path(f"{label}_{i}"), summary={"test_ce": ce, "test_best_ce": ce}, seed=s,
                   data_seed=d, data_seed_recorded=True) for i, (ce, (s, d)) in enumerate(zip(ces, PAIRS))]
    return Arm(label=label, group="g", runs=runs, **kwargs)


def test_inapplicable_capacity_clause_caps_the_verdict() -> None:
    manifest = Manifest(arms=[
        _arm(pa.EVALUATED, [3.00, 3.10, 3.05]), _arm(pt.DECISION_COMPARATOR, [3.10, 3.25, 3.12]),
        _arm(pt.DECISION_FROZEN, [3.10, 3.25, 3.12]),
        Arm(label=pt.DECISION_CAPACITY, group="g", inapplicable="no coordinate MLP")], selection="final")
    record = pt.decision_rule(manifest)
    assert record["clauses"]["a"]["holds"] is True and record["clauses"]["b"]["holds"] is True
    assert record["verdict"] == pt.DECISION_NO_CAPACITY_VERDICT == "undetermined (no capacity control on this base)"
    _, markdown, _ = pt.decision_rule_table(manifest)
    assert f"Claim: **{pt.DECISION_NO_CAPACITY_VERDICT}**" in markdown


# ---- item 44: excluded positions are not all overlap ---------------------------------------------

def test_methods_contract_splits_overlap_and_padding_or_stays_neutral() -> None:
    summary = {"test_scored_targets": 90, "test_expected_targets": 100, "test_excluded_targets": 10,
               "final_val_scored_targets": 45, "final_val_expected_targets": 50, "final_val_excluded_targets": 5}
    split = RunRecord(run_dir=Path("r"), summary={**summary, "test_overlap_context_targets": 7,
                                                  "test_padding_targets": 3})
    items = dict((label, str(value)) for label, value in pt._evaluation_items(split, "best") if label)
    assert "(7 overlapping-context and 3 padding positions excluded)" in items["test scored targets"]
    assert items["validation scored targets"].endswith("(5 excluded positions)")
    neutral = dict((label, str(value)) for label, value in pt._evaluation_items(RunRecord(run_dir=Path("r"),
                                                                                         summary=summary), "best")
                   if label)
    assert neutral["test scored targets"] == "90 of 100 window positions (10 excluded positions)"
    assert json.dumps(neutral).count("overlapping") == 0
