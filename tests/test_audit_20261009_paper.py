r"""Audit 2026-10-09 paper-pipeline items (paper_arms, vfe3.paper, vfe3.baseline).

Tiny synthetic bases and hand-written run directories only (K = 4 or 8, two heads, two layers);
nothing is trained. Covered: the tiers, code identity on reused and listed runs, the manifest flags
carried onto the arms and read by the decision rule, the paired-t p-value and interval, the comparator
resolved from the base's positional scheme, the compute-matched deferral, inert base values that
become live, the CPU-fallback guard, the linear-decoder arm, and the shared post-norm without the
attention residual.
"""

from __future__ import annotations

import json
import logging
import math
import warnings
from pathlib import Path
from typing import Any, Dict, Sequence

import pytest
import torch

import paper_arms as pa
import vfe3.paper.runs as pr
from tests.test_paper_arms_20260928 import (  # noqa: F401 - fixtures used by name
    _cpu_one_thread, _fake_run, _tiny_cfg, _write_base, caches,
)
from vfe3.paper import tables as pt
from vfe3.paper.runs import Arm, Manifest, RunRecord, resolve_manifest

LOGGER = logging.getLogger("audit-20261009-paper")
PAIRS = [(6, 3), (23, 23), (54, 54)]


# ---- tiers -------------------------------------------------------------------------------------

def test_tiers_schedule_the_frozen_clause_and_the_plain_matched_baseline() -> None:
    assert pa.DECISION_FROZEN in pa.TIER_B
    assert "Matched context-processing parameters" in pa.TIER_A
    assert "Linear decoder" not in (*pa.TIER_A, *pa.TIER_B)
    assert set(pa.TIER_A) <= set(pa.ARMS) and set(pa.TIER_B) <= set(pa.ARMS)


# ---- code identity -----------------------------------------------------------------------------

def _identity(run: Path, **provenance: Any) -> Path:
    path = run / "provenance.json"
    record = json.loads(path.read_text()) if path.is_file() else {}
    path.write_text(json.dumps({**record, **provenance}))
    return run


def test_code_identity_mismatch_compares_the_recorded_identity_with_the_tree(
        monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def current(kind: str) -> Dict[str, Any]:
        calls.append(kind)
        return ({"source_tree_sha256": "tree"} if kind == "source_tree_sha256"
                else {"git_sha": "abc", "git_dirty": False, "git_dirty_fingerprint": None})

    monkeypatch.setattr(pr, "current_code_identity", current)
    cache: Dict[str, Dict[str, Any]] = {}
    none = RunRecord(run_dir=Path("a"))
    assert pr.code_identity_mismatch(none, current=cache) == "the run records no code identity"
    same = RunRecord(run_dir=Path("b"), provenance={"git_sha": "abc", "git_dirty": False})
    assert pr.code_identity_mismatch(same, current=cache) is None
    other = RunRecord(run_dir=Path("c"), provenance={"git_sha": "434122c", "git_dirty": True,
                                                     "git_dirty_fingerprint": "f"})
    assert "434122c" in pr.code_identity_mismatch(other, current=cache)
    archived = RunRecord(run_dir=Path("d"), provenance={
        "git_sha": "old", "source_archive": {"available": True, "tree_sha256": "tree"}})
    assert pr.code_identity_mismatch(archived, current=cache) is None     # content hash wins over the sha
    assert calls == ["git", "source_tree_sha256"]                          # computed once per kind


def test_reused_base_run_with_other_code_is_noted_and_flagged_not_rejected(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pr, "current_code_identity",
                        lambda kind: {"git_sha": "head", "git_dirty": False, "git_dirty_fingerprint": None})
    run = _write_base(tmp_path, _tiny_cfg(seed=6))
    _identity(run, data_seed=3, git_sha="434122c", git_dirty=True, git_dirty_fingerprint="f")
    (run / "summary.json").write_text(json.dumps({"n_steps": 100, "wall_time_s": 20.0, "test_ce": 3.5}))
    base = pa.load_base("tiny", run)
    plans = pa.build_plan(base, budget="base_protocol", seeds=[(6, 3)], labels=[pa.EVALUATED], counts=False)
    assert plans[0].reused == [(6, 3)]
    assert any("code identity is not the current tree's" in note and "434122c" in note for note in plans[0].notes)
    exp = tmp_path / "exp"
    manifest = pa.build_manifest(exp, plans, base=base, budget="base_protocol", seeds=[(6, 3)])
    assert manifest["arms"][pa.EVALUATED]["runs"] == [str(base.run_dir)]             # kept
    flagged = manifest["paper_arms"]["rows"][pa.EVALUATED]["code_identity_mismatch"]
    assert [entry["run"] for entry in flagged] == [str(base.run_dir)] and "434122c" in flagged[0]["reason"]
    arm = resolve_manifest(json.loads(json.dumps(manifest))).arms[0]
    assert arm.code_identity_mismatch == flagged and arm.reuse_verified is True and arm.stale == []


# ---- the decision rule: flags and the paired-t report ------------------------------------------

def _arm(label: str, ces: Sequence[float], pairs: Sequence[tuple] = PAIRS, **flags: Any) -> Arm:
    runs = [RunRecord(run_dir=Path(f"{label[:8]}_{s}_{d}"), summary={"test_ce": ce, "n_steps": 10},
                      seed=s, data_seed=d) for ce, (s, d) in zip(ces, pairs)]
    return Arm(label=label, group="g", runs=runs, **flags)


def _manifest(**flags: Dict[str, Any]) -> Manifest:
    return Manifest(arms=[
        _arm(pa.EVALUATED, [3.00, 3.10, 3.05], **flags.get("evaluated", {})),
        _arm(pt.DECISION_FROZEN, [3.10, 3.25, 3.12], **flags.get("frozen", {})),
        _arm(pt.DECISION_COMPARATOR, [3.10, 3.25, 3.12], **flags.get("comparator", {})),
        _arm(pt.DECISION_CAPACITY, [3.10, 3.25, 3.12], **flags.get("capacity", {})),
    ], selection="final")


def test_decision_rule_reports_the_one_sided_paired_t_p_value_and_interval() -> None:
    record = pt.decision_rule(_manifest())
    a = record["clauses"]["a"]
    d = [0.10, 0.15, 0.07]
    mean = sum(d) / 3
    se = math.sqrt(sum((x - mean) ** 2 for x in d) / 2) / math.sqrt(3)
    t = mean / se
    # Student t on 2 degrees of freedom has the closed form F(t) = 1/2 + t / (2 sqrt(2 + t^2)), and
    # its quantile is t_q = (2q - 1) / sqrt(2 q (1 - q)).
    assert a["t"] == pytest.approx(t)
    assert a["p_one_sided"] == pytest.approx(0.5 - t / (2 * math.sqrt(2 + t * t)), rel=1e-9)
    t975 = 0.95 / math.sqrt(2 * 0.975 * 0.025)
    assert a["ci95"] == pytest.approx([mean - t975 * se, mean + t975 * se], rel=1e-9)
    assert a["holds"] is True and record["verdict"] == "stands"           # the registered rule is unchanged
    latex, markdown, _ = pt.decision_rule_table(_manifest())
    assert "One-sided paired-t p" in markdown and f"{a['p_one_sided']:.4f}" in markdown
    assert all(line.count("&") == 5 for line in latex.splitlines() if line.endswith(r"\\"))


@pytest.mark.parametrize(("who", "flags", "fragment"), [
    ("comparator", {"reuse_verified": False}, "reuse_verified false"),
    ("comparator", {"stale": [{"run": "x", "reason": "lr"}]}, "stale"),
    ("comparator", {"homogeneity": {"homogeneous": False, "violations": [{"field": "lr"}]}}, "not homogeneous"),
])
def test_a_flagged_arm_makes_its_clauses_undetermined(who: str, flags: Dict[str, Any], fragment: str) -> None:
    record = pt.decision_rule(_manifest(**{who: flags}))
    a, b = record["clauses"]["a"], record["clauses"]["b"]
    assert a["holds"] is None and any(fragment in reason for reason in a["undetermined"])
    assert a["n"] == 3 and a["mean"] is not None                          # statistics still reported
    if who == "evaluated":                                                # the evaluated arm is in both clauses
        assert b["holds"] is None and record["verdict"] == "undetermined"
    else:
        assert b["holds"] is True and not b["undetermined"] and record["verdict"] == "undetermined"
    _, markdown, _ = pt.decision_rule_table(_manifest(**{who: flags}))
    assert "Clause (a) undetermined" in markdown and "| undetermined |" in markdown


def test_resolve_manifest_carries_the_paper_arms_row_flags(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "summary.json").write_text(json.dumps({"test_ce": 3.0}))
    manifest = {"arms": {"A": [str(run)], "B": []},
                "paper_arms": {"rows": {"A": {"reuse_verified": False, "stale": [{"run": "s", "reason": "r"}],
                                              "code_identity_mismatch": [{"run": str(run), "reason": "c"}]}}}}
    a, b = resolve_manifest(manifest).arms
    assert (a.reuse_verified, a.stale, a.code_identity_mismatch) == (
        False, [{"run": "s", "reason": "r"}], [{"run": str(run), "reason": "c"}])
    assert (b.reuse_verified, b.stale, b.code_identity_mismatch) == (None, [], [])


# ---- the comparator from the base's positional scheme ------------------------------------------

def test_comparator_follows_the_positional_scheme_and_the_alibi_duplicate_is_inapplicable(
        tmp_path: Path) -> None:
    import train_baseline  # noqa: F401
    from vfe3.baseline import BaselineConfig

    alibi = pa.load_base("alibi", _write_base(tmp_path / "a", _tiny_cfg(beta_attention_prior="causal_alibi",
                                                                         rope_mode="none")))
    assert pa.decision_comparator(alibi.cfg) == pa.DECISION_COMPARATOR_ALIBI == pt.DECISION_COMPARATOR
    plans = {p.label: p for p in pa.build_plan(alibi, budget="base_protocol", seeds=[(1, 1)], counts=False,
                                               labels=list(pa.TIER_A), baseline_overrides={"lr": 3e-3})}
    frame_size = plans[pa.DECISION_COMPARATOR_MATCHED]
    assert frame_size.status == "inapplicable" and "causal_alibi" in frame_size.reason
    # Its config would be the as-run arm's: resolve both specs without the inapplicability.
    as_run = plans[pa.DECISION_COMPARATOR_ALIBI]
    spec = {key: value for key, value in pa.ARMS[pa.DECISION_COMPARATOR_MATCHED].items() if key != "inapplicable"}
    duplicate = pa.resolve_arm(alibi, pa.DECISION_COMPARATOR_MATCHED, spec, budget="base_protocol",
                               seeds=[(1, 1)], counts=False, baseline_overrides={"lr": 3e-3})
    assert as_run.status == duplicate.status == "ready" and duplicate.cfg == as_run.cfg
    assert isinstance(duplicate.cfg, BaselineConfig)
    manifest = pa.build_manifest(tmp_path / "exp", list(plans.values()), base=alibi, budget="base_protocol",
                                 seeds=[(1, 1)])
    assert manifest["decision"]["comparator"] == pa.DECISION_COMPARATOR_ALIBI
    assert resolve_manifest(json.loads(json.dumps(manifest))).decision["comparator"] == pa.DECISION_COMPARATOR_ALIBI

    for name, over in (("rope", {"beta_attention_prior": "causal_alibi", "rope_mode": "frame_value"}),
                       ("causal", {"beta_attention_prior": "causal", "rope_mode": "none"})):
        base = pa.load_base(name, _write_base(tmp_path / name, _tiny_cfg(**over)))
        assert pa.decision_comparator(base.cfg) == pa.DECISION_COMPARATOR_MATCHED
        plan = pa.build_plan(base, budget="base_protocol", seeds=[(1, 1)], counts=False,
                             labels=[pa.DECISION_COMPARATOR_MATCHED])[0]
        assert plan.status == "ready"
        written = pa.build_manifest(tmp_path / f"exp_{name}", [plan], base=base, budget="base_protocol",
                                    seeds=[(1, 1)])
        assert written["decision"]["comparator"] == pa.DECISION_COMPARATOR_MATCHED
        assert pt.decision_rule(resolve_manifest(json.loads(json.dumps(written))))["labels"]["comparator"] \
            == pa.DECISION_COMPARATOR_MATCHED
    # A manifest without a decision record falls back to the registered constant.
    assert pt.decision_rule(Manifest())["labels"]["comparator"] == pt.DECISION_COMPARATOR


# ---- compute-matched deferral ------------------------------------------------------------------

def test_compute_matched_rule_waits_for_every_planned_evaluated_seed_pair(
        tmp_path: Path, caches: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vfe3.baseline  # noqa: F401
    import train_baseline as tb

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a deferred rule calibrates nothing")

    monkeypatch.setattr(tb, "calibrate_widths", refuse)
    monkeypatch.setitem(pa.BUDGETS, "tiny", lambda cfg, dataset: {
        "max_steps": 4, "eval_interval": 2, "checkpoint_interval": 4})
    run = _identity(_write_base(tmp_path, _tiny_cfg(seed=1, max_steps=40)), data_seed=1)
    (run / "summary.json").write_text(json.dumps({"n_steps": 100, "wall_time_s": 20.0, "test_ce": 3.5}))
    base = pa.load_base("tiny", run)
    seeds = [(1, 1), (2, 2)]
    plans = {p.label: p for p in pa.build_plan(base, budget="tiny", seeds=seeds, counts=False,
                                               labels=[pa.EVALUATED, "Matched measured compute"])}
    plan = plans["Matched measured compute"]
    exp = pa.experiment_dir(tmp_path / "out", "tiny", "tiny")
    rule_path = exp / plan.slug / pa.WIDTH_RULE_FILE

    # One of two planned pairs complete: deferred, nothing frozen.
    _fake_run(exp / "evaluated", "3.21_evaluated_s2_d2", plans[pa.EVALUATED].cfg, seed=2, data_seed=2)
    assert pa.compute_matched_config(plan, exp, calibration_steps=4, width_ladder=(1, 2), logger=LOGGER) is None
    assert not rule_path.exists()
    # The base run fills (1, 1) through the fallback, but its config is not the plan's (the base's
    # max_steps 40 is not the tiny budget's 4): still deferred, with the reason.
    assert pa.completed_run(exp, pa.EVALUATED, 1, 1, base=base, budget="tiny") is None   # tiny is not base_protocol
    monkeypatch.setitem(pa.BUDGETS, "base_protocol", pa.BUDGETS["tiny"])
    found = pa.completed_run(exp, pa.EVALUATED, 1, 1, base=base, budget="base_protocol")
    assert found == base.run_dir
    assert pa.reuse_mismatch(found, plan.matched_cfg, dataset=plan.dataset, seed=1, data_seed=1) is not None
    stale: list = []
    assert pa.completed_run(exp, pa.EVALUATED, 1, 1, base=base, budget="base_protocol", cfg=plan.matched_cfg,
                            dataset=plan.dataset, stale=stale) is None                # checked: rejected
    assert [entry["run"] for entry in stale] == [str(base.run_dir)] and stale[0]["reason"]
    assert pa.compute_matched_config(plan, exp, calibration_steps=4, width_ladder=(1, 2), logger=LOGGER,
                                     base=base, budget="base_protocol") is None
    assert not rule_path.exists()


# ---- inert base values, the CPU guard, the linear decoder --------------------------------------

def test_an_inert_base_value_that_becomes_live_is_noted_with_its_value(tmp_path: Path) -> None:
    base = pa.load_base("tiny", _write_base(tmp_path, _tiny_cfg(m_per_type_hidden_bias_lr=0.0025, m_phi_lr=0.003)))
    assert "m_per_type_hidden_bias_lr" in base.cfg.inactive_fields()
    plan = pa.build_plan(base, budget="base_protocol", seeds=[(1, 1)], counts=False,
                         labels=["Identity frames, per-type additive table of equal size"])[0]
    assert plan.status == "ready" and plan.cfg.m_per_type_hidden_bias_lr == 0.0025
    assert "m_per_type_hidden_bias_lr=0.0025 is inert on the base and live in this arm" in plan.notes


def test_training_launch_refuses_a_silent_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = {**pa.CONFIG, "bases": [], "dry_run": False, "device": None}
    with pytest.raises(SystemExit, match="CUDA is not available"):
        pa.main(config, bases={})
    assert pa.main({**config, "device": "cpu"}, bases={}) == 0           # a deliberate CPU launch
    assert pa.main({**config, "dry_run": True}, bases={}) == 0           # a dry run trains nothing


def test_linear_decoder_arm_is_one_change_and_inapplicable_on_a_linear_base(tmp_path: Path) -> None:
    from tests.test_paper_arms_20260928 import _paper_artifacts_rows

    assert _paper_artifacts_rows()["Linear decoder"] == "_RETRAINING"
    linear = pa.load_base("linear", _write_base(tmp_path / "l", _tiny_cfg(decode_mode="linear")))
    plan = pa.build_plan(linear, budget="base_protocol", seeds=[(1, 1)], counts=False, labels=["Linear decoder"])[0]
    assert plan.status == "inapplicable" and "decode_mode='linear'" in plan.reason
    kl = pa.load_base("kl", _write_base(tmp_path / "k", _tiny_cfg(decode_mode="diagonal_chunked")))
    plan = pa.build_plan(kl, budget="base_protocol", seeds=[(1, 1)], labels=["Linear decoder"])[0]
    assert plan.status == "ready" and plan.diff == {"decode_mode": ("diagonal_chunked", "linear")}
    assert plan.params["total"] > 0


# ---- shared post-norm without the attention residual -------------------------------------------

def _vfe_base(**over: Any) -> Any:
    from tests.profiles import test_config

    values: Dict[str, Any] = dict(
        vocab_size=11, embed_dim=8, n_heads=2, batch_size=2, max_seq_len=6, n_layers=2, n_e_steps=1,
        use_block_mlp=True, block_mlp_expansion=2, family="gaussian_diagonal", gauge_group="block_glk",
        attention_mode="projected_predictive", decode_mode="linear", generate_embedding_figures=False,
        pos_phi="learned", norm_type_block="layernorm", layernorm_affine=True, skip_belief_sigma_update=True)
    values.update(over)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return test_config(**values)


def test_post_shared_baseline_drops_the_attention_residual_of_a_base_without_one() -> None:
    from vfe3.baseline import BaselineConfig, BaselineTransformer, _fp32_layer_norm

    with_residual = BaselineConfig.matched_to(_vfe_base(attention_residual=True), norm_placement="post_shared")
    without = BaselineConfig.matched_to(_vfe_base(attention_residual=False), norm_placement="post_shared")
    assert with_residual.attention_residual is True and without.attention_residual is False
    assert BaselineConfig.matched_to(_vfe_base(attention_residual=False)).attention_residual is True   # pre-LN
    assert "attention_residual" in BaselineConfig.matched_to(_vfe_base()).inactive_fields()
    with pytest.raises(ValueError, match="post_shared"):
        BaselineConfig.matched_to(_vfe_base(), attention_residual=False)
    assert BaselineConfig.from_serialized(without.serialize()) == without
    legacy = {k: v for k, v in with_residual.serialize().items() if k != "attention_residual"}
    assert BaselineConfig.from_serialized(legacy).attention_residual is True           # saved before the field

    torch.manual_seed(0)
    model_a = BaselineTransformer(with_residual)
    torch.manual_seed(0)
    model_b = BaselineTransformer(without)
    assert [n for n, _ in model_a.named_parameters()] == [n for n, _ in model_b.named_parameters()]
    block = model_b.blocks[0]
    h = torch.randn(2, 6, 8, generator=torch.Generator().manual_seed(1))
    prior = torch.zeros(2, 6, 6)                                                       # (H, N, N)
    with torch.no_grad():
        a = block.attn(h, prior, None)
        expected = _fp32_layer_norm(model_b.shared_norm, a)
        expected = expected + block.mlp(expected)
        out = block(h, prior, None, shared_norm=model_b.shared_norm)
        torch.testing.assert_close(out, expected, rtol=0., atol=0.)
        residual = model_a.blocks[0](h, prior, None, shared_norm=model_a.shared_norm)
        assert not torch.allclose(out, residual)


def test_paper_arms_post_norm_arm_matches_a_base_without_the_residual_and_says_so(tmp_path: Path) -> None:
    import train_baseline  # noqa: F401
    label = "Per-type read table, ALiBi, shared post-norm"
    for residual in (True, False):
        base = pa.load_base("tiny", _write_base(tmp_path / str(residual), _tiny_cfg(attention_residual=residual)))
        plan = pa.build_plan(base, budget="base_protocol", seeds=[(1, 1)], counts=False, labels=[label])[0]
        # Round 2 #3: the arm keeps the residual on a base without one, as a declared difference.
        assert plan.status == "ready" and plan.cfg.attention_residual is True
        assert any("attention residual kept" in note for note in plan.notes) is (not residual)


def test_a_code_identity_mismatch_is_reported_without_changing_the_verdict() -> None:
    flags = {"code_identity_mismatch": [{"run": "x", "reason": "r"}]}
    record = pt.decision_rule(_manifest(evaluated=flags))
    a, b = record["clauses"]["a"], record["clauses"]["b"]
    assert a["holds"] is True and b["holds"] is True and record["verdict"] == "stands"
    assert not a["undetermined"] and any("code_identity_mismatch" in n for n in a["code_identity_notes"])
    _, markdown, _ = pt.decision_rule_table(_manifest(evaluated=flags))
    assert "Clause (a) note:" in markdown and "| undetermined |" not in markdown


# ---- clause (c): the equal-size per-type table (amendment of October 9, 2026) -----------------

def _capacity_manifest(capacity: Sequence[float], pairs: Sequence[tuple] = PAIRS, **capacity_flags: Any) -> Manifest:
    return Manifest(arms=[
        _arm(pa.EVALUATED, [3.00, 3.10, 3.05]),
        _arm(pt.DECISION_FROZEN, [3.10, 3.25, 3.12]),
        _arm(pt.DECISION_COMPARATOR, [3.10, 3.25, 3.12]),
        _arm(pt.DECISION_CAPACITY, capacity, pairs, **capacity_flags),
    ], selection="final")


def test_capacity_label_is_shared_and_written_into_the_manifest_decision(tmp_path: Path) -> None:
    assert pa.DECISION_CAPACITY == pt.DECISION_CAPACITY == "Identity frames, per-type additive table of equal size"
    assert pa.DECISION_CAPACITY in pa.ARMS
    base = pa.load_base("tiny", _write_base(tmp_path, _tiny_cfg()))
    plans = pa.build_plan(base, budget="base_protocol", seeds=[(1, 1)], counts=False, labels=[pa.EVALUATED])
    manifest = pa.build_manifest(tmp_path / "exp", plans, base=base, budget="base_protocol", seeds=[(1, 1)])
    assert manifest["decision"]["capacity"] == pa.DECISION_CAPACITY
    resolved = resolve_manifest(json.loads(json.dumps(manifest)))
    assert resolved.decision["capacity"] == pa.DECISION_CAPACITY
    assert pt.decision_rule(resolved)["labels"]["capacity"] == pa.DECISION_CAPACITY
    # The manifest's label wins; the constant is the fallback for a manifest without one.
    custom = Manifest(arms=[], decision={"capacity": "Some other table"})
    assert pt.decision_rule(custom)["labels"]["capacity"] == "Some other table"
    assert pt.decision_rule(Manifest())["labels"]["capacity"] == pt.DECISION_CAPACITY


def test_clause_c_holds_and_the_claim_stands_only_when_every_clause_holds() -> None:
    record = pt.decision_rule(_capacity_manifest([3.10, 3.25, 3.12]))
    c = record["clauses"]["c"]
    assert c["arm"] == pt.DECISION_CAPACITY and c["n"] == 3 and c["holds"] is True
    assert c["margin"] == pytest.approx(record["clauses"]["a"]["margin"])
    assert record["verdict"] == "stands"


def test_clause_c_failing_retracts_the_claim_although_a_and_b_hold() -> None:
    manifest = _capacity_manifest([3.01, 3.10, 3.07])                          # d = 0.01, 0.00, 0.02
    record = pt.decision_rule(manifest)
    assert record["clauses"]["a"]["holds"] is True and record["clauses"]["b"]["holds"] is True
    assert record["clauses"]["c"]["holds"] is False and record["verdict"] == "retracted"
    latex, markdown, _ = pt.decision_rule_table(manifest)
    assert "(c)" in latex and "equal-size per-type additive table" in markdown
    rows = [line for line in latex.splitlines() if line.endswith(r"\\")]
    assert len(rows) == 4 and all(line.count("&") == 5 for line in rows)        # header and three clauses
    assert "Claim: **retracted**" in markdown and "October 9, 2026" in markdown


def test_clause_c_is_strict_like_clause_a_while_b_is_not(monkeypatch: pytest.MonkeyPatch) -> None:
    # Identical differences of 0.5 nats give SE = 0 and the margin max(0, 0.5) = 0.5 exactly.
    monkeypatch.setattr(pt, "DECISION_MARGIN_NATS", 0.5)
    manifest = Manifest(arms=[
        _arm(pa.EVALUATED, [3.0, 3.0, 3.0]),
        _arm(pt.DECISION_FROZEN, [3.5, 3.5, 3.5]),
        _arm(pt.DECISION_COMPARATOR, [3.5, 3.5, 3.5]),
        _arm(pt.DECISION_CAPACITY, [3.5, 3.5, 3.5]),
    ], selection="final")
    clauses = pt.decision_rule(manifest)["clauses"]
    assert {k: (c["mean"], c["margin"]) for k, c in clauses.items()} == {k: (0.5, 0.5) for k in "abc"}
    assert (clauses["a"]["holds"], clauses["b"]["holds"], clauses["c"]["holds"]) == (False, True, False)


def test_an_inapplicable_capacity_arm_is_na_and_does_not_block_the_verdict() -> None:
    reason = "the per-type hidden bias needs the base's coordinate block MLP"
    manifest = _capacity_manifest([], inapplicable=reason)
    record = pt.decision_rule(manifest)
    c = record["clauses"]["c"]
    assert c["holds"] is None and c["inapplicable"] == reason and c["n"] == 0
    assert record["verdict"] == pt.DECISION_NO_CAPACITY_VERDICT      # (a) and (b) alone cannot stand (round 2 #27)
    latex, markdown, _ = pt.decision_rule_table(manifest)
    row = next(line for line in latex.splitlines() if line.startswith("(c)"))
    assert row.count("&") == 5 and row.endswith(f"& {pt.NOT_APPLICABLE} \\\\")
    assert f"Clause (c) not applicable: {reason}" in markdown
    # Inapplicability never excuses clauses (a) or (b): an inapplicable comparator stays undetermined.
    no_comparator = Manifest(arms=[
        _arm(pa.EVALUATED, [3.00, 3.10, 3.05]), _arm(pt.DECISION_FROZEN, [3.10, 3.25, 3.12]),
        Arm(label=pt.DECISION_COMPARATOR, group="g", inapplicable="x"),
        _arm(pt.DECISION_CAPACITY, [3.10, 3.25, 3.12])], selection="final")
    assert pt.decision_rule(no_comparator)["verdict"] == "undetermined"
    # A failing applicable clause still retracts beside an inapplicable (c).
    failing = _capacity_manifest([], inapplicable=reason)
    failing.arms[1] = _arm(pt.DECISION_FROZEN, [3.01, 3.10, 3.07])
    assert pt.decision_rule(failing)["verdict"] == "retracted"


def test_an_applicable_capacity_arm_without_pairs_is_undetermined() -> None:
    record = pt.decision_rule(_capacity_manifest([3.10], PAIRS[:1]))
    assert record["clauses"]["c"]["holds"] is None and record["verdict"] == "undetermined"
    flagged = pt.decision_rule(_manifest(capacity={"reuse_verified": False}))
    assert flagged["clauses"]["c"]["holds"] is None and flagged["clauses"]["c"]["undetermined"]
    assert flagged["verdict"] == "undetermined"
    absent = Manifest(arms=[_arm(pa.EVALUATED, [3.00, 3.10, 3.05]), _arm(pt.DECISION_FROZEN, [3.10, 3.25, 3.12]),
                            _arm(pt.DECISION_COMPARATOR, [3.10, 3.25, 3.12])], selection="final")
    record = pt.decision_rule(absent)
    assert record["verdict"] == "undetermined" and any("capacity" in p for p in record["problems"])


def test_paper_arms_writes_an_inapplicable_capacity_arm_that_the_rule_reads_as_na(tmp_path: Path) -> None:
    base = pa.load_base("tiny", _write_base(tmp_path, _tiny_cfg(use_block_mlp=False)))
    plans = pa.build_plan(base, budget="base_protocol", seeds=[(1, 1)], counts=False,
                          labels=[pa.EVALUATED, pa.DECISION_CAPACITY])
    manifest = pa.build_manifest(tmp_path / "exp", plans, base=base, budget="base_protocol", seeds=[(1, 1)])
    assert "coordinate block MLP" in manifest["arms"][pa.DECISION_CAPACITY]["inapplicable"]
    record = pt.decision_rule(resolve_manifest(json.loads(json.dumps(manifest))))
    assert record["clauses"]["c"]["inapplicable"]


# ---- the total-parameter-matched conventional transformer -------------------------------------

TOTAL_MATCHED = "Matched total parameters"


def _baseline_total(K: int, *, L: int, e: int, V: int) -> int:
    r"""Pre-LN baseline without a per-type table: :math:`L(4K^2 + 2eK^2 + eK + 5K) + 2K + 2VK + V`."""
    return L * (4 * K * K + 2 * e * K * K + e * K + 5 * K) + 2 * K + 2 * V * K + V


def test_total_matched_arm_is_a_baseline_row_in_table_order_and_in_no_tier() -> None:
    from tests.test_paper_arms_20260928 import _paper_artifacts_rows

    spec = pa.ARMS[TOTAL_MATCHED]
    assert spec["group"] == pa._BASELINES and spec["slug"] == "tf_matched_total"
    assert spec["baseline"]["per_type_table"] == "none" and spec["baseline"]["total_matched"] is True
    assert _paper_artifacts_rows()[TOTAL_MATCHED] == "_BASELINES"
    labels = list(pa.ARMS)
    assert labels.index("Matched measured compute") + 1 == labels.index(TOTAL_MATCHED)
    assert TOTAL_MATCHED not in (*pa.TIER_A, *pa.TIER_B)


def test_total_parameter_width_picks_the_nearest_whole_head_width() -> None:
    import dataclasses

    from vfe3.baseline import BaselineConfig, total_parameter_width

    target = 233_557_505                                      # the 32.65 base (K = 128, 6 layers, V = 50257)
    cfg = BaselineConfig(vocab_size=50257, embed_dim=128, n_heads=4, n_layers=6, mlp_expansion=8)
    assert cfg.total_parameters() == _baseline_total(128, L=6, e=8, V=50257)
    rule = total_parameter_width(cfg, target)
    chosen = rule["config"]
    K = chosen.embed_dim
    assert K % 32 == 0 and chosen.n_heads == K // 32 and chosen.n_layers == 6
    assert rule["total_parameters"] == chosen.total_parameters() == _baseline_total(K, L=6, e=8, V=50257)
    for other in (K - 32, K + 32):                            # no other whole-head width is closer
        assert abs(_baseline_total(other, L=6, e=8, V=50257) - target) >= abs(rule["total_parameters"] - target)
    assert abs(rule["relative_error"]) < 0.03
    assert rule["target_total"] == target and rule["head_dim"] == 32 and rule["width"] == K
    assert total_parameter_width(cfg, target) == rule                       # deterministic
    assert all(getattr(cfg, f.name) == getattr(chosen, f.name)
               for f in dataclasses.fields(cfg) if f.name not in ("embed_dim", "n_heads", "hidden_lr_scale"))
    assert chosen.hidden_lr_scale == rule["hidden_lr_scale"] == 128 / K           # round 2 #18
    with pytest.raises(ValueError):
        total_parameter_width(cfg, 0)


def test_total_matched_arm_matches_the_base_total_and_declares_only_the_width(tmp_path: Path) -> None:
    import dataclasses

    import train_baseline  # noqa: F401
    from vfe3.baseline import total_parameter_width

    # A vocabulary large enough that the frame table makes the base wider in total than its dense twin.
    base = pa.load_base("tiny", _write_base(tmp_path, _tiny_cfg(vocab_size=200)))
    plans = {p.label: p for p in pa.build_plan(base, budget="base_protocol", seeds=[(1, 1)],
                                               labels=[pa.EVALUATED, TOTAL_MATCHED],
                                               baseline_overrides={"lr": 3e-3})}
    plan, evaluated = plans[TOTAL_MATCHED], plans[pa.EVALUATED]
    assert plan.status == "ready" and plan.kind == "baseline"
    assert set(plan.diff) == {"embed_dim", "n_heads", "hidden_lr_scale"}         # round 2 #18
    assert plan.cfg.n_layers == evaluated.cfg.n_layers and plan.cfg.max_steps == evaluated.cfg.max_steps
    d = evaluated.cfg.embed_dim // evaluated.cfg.n_heads
    assert plan.cfg.embed_dim % d == 0 and plan.cfg.embed_dim > evaluated.cfg.embed_dim
    target = evaluated.params["total"]
    assert plan.params["total"] == plan.cfg.total_parameters()
    at_base_width = dataclasses.replace(plan.cfg, embed_dim=int(evaluated.cfg.embed_dim),
                                        n_heads=int(evaluated.cfg.n_heads), hidden_lr_scale=1.0)
    assert plan.cfg == total_parameter_width(at_base_width, target)["config"]
    assert any("total parameters" in note and f"{target:,}" in note for note in plan.notes)
    # The sizing does not depend on whether the plan reports counts.
    quiet = pa.build_plan(base, budget="base_protocol", seeds=[(1, 1)], labels=[TOTAL_MATCHED], counts=False,
                          baseline_overrides={"lr": 3e-3})[0]
    assert quiet.cfg == plan.cfg
