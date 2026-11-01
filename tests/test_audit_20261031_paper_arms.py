r"""Audit 2026-10-31, package paper-arms: the planning, reuse and manifest logic of ``paper_arms.py``.

paper-arms-harness-1: a subset launch rebuilt ``m2_manifest.json`` from its own selection and seeds, so the decision
rows lost their verification and seed pairs and the registered rule became undetermined; unselected arms are now
resolved and verified, and the manifest keeps the seed pairs it already records. paper-arms-harness-2: an arm whose
every change is inert on the base trained a roundoff replicate of the evaluated arm; it is inapplicable now, and every
manifest row carries its plan notes. paper-arms-harness-5: a base that already is an arm's configuration crashed the
planner. paper-arms-harness-7 and gap-paper-arm-offlive-routes-6: the frame-free Cholesky-table arm was invalid on a
Fisher-table base; it declares the AdamW table step, and the covariant stack shares it. gap-paper-arm-offlive-routes-2:
on a base with positional frames the Cholesky arm is not the covariant stack's equivalent, and the identity-frame rows
lose the positional channel. gap-paper-arm-offlive-routes-9: the intertwiner row's mean-scale change.
paper-arms-harness-8 and -9: base reuse under trust_resume_checkpoint and the recorded data identity.
paper-arms-harness-10: an interrupted run resumes from its last checkpoint. paper-arms-harness-11: word-level rows
under one budget directory. paper-arms-harness-12: baseline rows' defining choices in the manifest.
paper-arms-harness-14: plan totals. gap-mixed-precision-routes-2: effective precision under fp16.

Every config is tiny (K = 4, two heads, two layers, vocabulary 7); training is replaced by writers of complete run
folders, except a few 2- and 4-step CPU runs: one pins the data-identity helper to the provenance recorder, and the
resume tests compare an interrupted and resumed run with an uninterrupted one bitwise.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch

import paper_arms as pa
import vfe3.data.datasets as datasets_mod
import vfe3.experiment as experiment
import vfe3.paper.runs as pr
from tests.test_paper_arms_20260928 import (  # noqa: F401 - fixtures used by name
    BPE, WORD, _cpu_one_thread, _fake_run, _tiny_cfg, _write_base, caches,
)
from vfe3.config import serialize_config

LOGGER = logging.getLogger("audit-20261031-paper-arms")
BLOCK_LEARNED = "Head-block family, learned frames"
BLOCK_CHOLESKY = "Head-block family, frame-free Cholesky table"
BLOCK_REFERENCE = "Head-block family, normalization, MLP and decoder in reference coordinates"
INTERTWINER = "Intertwiner-constrained nonlinear block"
IDENTITY_DECAYED = "Identity frames, decayed stored covariance"
IDENTITY_NO_DECAY = "Identity frames, stored covariance without decay"
REFERENCE_BLOCK = "LayerNorm and MLP in reference coordinates"
MLP_REMOVED = "MLP removed"
COMPUTE = "Matched measured compute"
DECISION = [pa.EVALUATED, pa.DECISION_COMPARATOR_ALIBI, pa.DECISION_FROZEN, pa.DECISION_CAPACITY]
PAIRS = [(6, 3), (23, 23), (54, 54)]
# A Fisher-table base (the live recipe's table step): a prior-bank decode with an untied decode bank.
FISHER = dict(m_table_update_mode="fisher", decode_mode="diagonal_chunked", untie_decode_bank=True, pos_phi="none")


@pytest.fixture(autouse=True)
def _fixed_code_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pr, "current_code_identity",
                        lambda kind: {"git_sha": "head", "git_dirty": False, "git_dirty_fingerprint": None})


@pytest.fixture(autouse=True)
def _no_user_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r"""No test reads the user's token cache: an empty cache directory unless the test asks for ``caches``."""
    empty = tmp_path / "empty_cache"
    empty.mkdir()
    monkeypatch.setattr(datasets_mod, "default_cache_dir", lambda: empty)


def _quiet(fn: Any, *args: Any, **kwargs: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def _base(root: Path, name: str = "tiny", *, provenance: Optional[Dict[str, Any]] = None, **over: Any) -> pa.Base:
    r"""A complete base run at model seed 6 and data seed 3 (the paper bases' pair)."""
    run = _write_base(root / name, _tiny_cfg(**{"seed": 6, **over}))
    (run / "provenance.json").write_text(json.dumps({"seed": 6, "data_seed": 3, **(provenance or {})}))
    (run / "summary.json").write_text(json.dumps({
        "n_steps": 100, "wall_time_s": 20.0, "test_ce": 3.0, "test_best_ce": 3.0, "best_step": 4}))
    return _quiet(pa.load_base, name, run)


def _plan(base: pa.Base, label: str, *, seeds: Any = ((6, 3),), counts: bool = False, **kwargs: Any) -> pa.ArmPlan:
    return _quiet(pa.build_plan, base, budget="base_protocol", seeds=list(seeds), labels=[label], counts=counts,
                  baseline_overrides={"lr": 3e-3}, **kwargs)[0]


# ---- fake training: complete run folders with chosen test CEs ----------------------------------

CE = {"evaluated": 3.000, "tf_type_table_alibi": 3.400, "frozen_frames": 3.250, "identity_frames_type_bias": 3.200}
JITTER = {1: 0.002, 6: 0.000, 23: 0.010, 54: 0.004}


def _write_run(folder: Path, config: Dict[str, Any], dataset: str, seed: int, data_seed: int, ce: float) -> Path:
    folder.mkdir(parents=True)
    (folder / "config.json").write_text(json.dumps({"config": config, "dataset": dataset}))
    (folder / "provenance.json").write_text(json.dumps({"seed": seed, "data_seed": data_seed}))
    (folder / "summary.json").write_text(json.dumps({
        "test_ce": ce, "test_best_ce": ce, "best_val_ppl": math.exp(ce + 0.01), "best_step": 4, "n_steps": 4,
        "evaluated_step": 4, "evaluation_weights": "final", "reloaded_best": False, "wall_time_s": 1.0}))
    return folder


def _fake_trainers(monkeypatch: pytest.MonkeyPatch) -> List[str]:
    r"""Replace ``run_vfe`` and ``run_baseline`` by writers of complete runs; return the list of trained labels."""
    import train_baseline

    trained: List[str] = []

    def ce(label: str, seed: int) -> float:
        return CE.get(label.rsplit("_s", 1)[0], 3.100) + JITTER.get(seed, 0.0)

    def run_vfe(cfg: Any, *, dataset: str, out_root: Any, seed: int, data_seed: int, label: str,
                device: Any = None, logger: Any = None, **kwargs: Any) -> Path:
        cfg = experiment._without_resume(experiment._with_seed(cfg, seed), label, LOGGER)
        trained.append(label)
        value = ce(label, seed)
        return _write_run(Path(out_root) / f"{math.exp(value):.2f}_{label}", serialize_config(cfg), dataset,
                          seed, data_seed, value)

    def run_baseline(cfg: Any, *, dataset: str, out_root: Any, seed: int, data_seed: int, label: str,
                     device: Any = None, logger: Any = None) -> Path:
        cfg = dataclasses.replace(cfg, seed=int(seed))
        trained.append(label)
        value = ce(label, seed)
        return _write_run(Path(out_root) / f"{math.exp(value):.2f}_{label}", cfg.serialize(), dataset, seed,
                          data_seed, value)

    monkeypatch.setattr(experiment, "run_vfe", run_vfe)
    monkeypatch.setattr(train_baseline, "run_baseline", run_baseline)
    return trained


def _launch(base: pa.Base, out: Path, labels: List[str], seeds: List[Tuple[int, int]],
            budget: str = "base_protocol") -> Dict[str, Any]:
    plans = _quiet(pa.build_plan, base, budget=budget, seeds=seeds, labels=labels, counts=False,
                   baseline_overrides={"lr": 3e-3})
    path = _quiet(pa.run_plan, base, plans, budget=budget, out_root=out, device="cpu", logger=LOGGER)
    return json.loads(path.read_text(encoding="utf-8"))


def _verdict(manifest: Dict[str, Any]) -> Dict[str, Any]:
    from vfe3.paper.runs import resolve_manifest
    from vfe3.paper.tables import decision_rule

    return _quiet(decision_rule, _quiet(resolve_manifest, manifest))


# ---- paper-arms-harness-1: a subset launch adds rows instead of replacing them -------------------

def test_a_subset_launch_keeps_the_decision_rows_and_the_seed_pairs_of_the_full_launch(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = _base(tmp_path, sigma_weight_decay=0.0, rope_mode="none", beta_attention_prior="causal_alibi")
    assert pa.decision_comparator(base.cfg) == pa.DECISION_COMPARATOR_ALIBI
    trained = _fake_trainers(monkeypatch)
    out = tmp_path / "out"
    full = _launch(base, out, DECISION, PAIRS)
    assert len(trained) == 11                                      # (6, 3) of the evaluated arm is the base run
    first = _verdict(full)
    assert first["verdict"] == "stands" and all(c["n"] == 3 for c in first["clauses"].values())

    def decision_rows(manifest: Dict[str, Any]) -> Dict[str, Tuple[Any, ...]]:
        shown = {pa.EVALUATED: pa.evaluated_label(base.cfg)}
        rows = manifest["paper_arms"]["rows"]
        return {label: (rows[shown.get(label, label)]["complete"], rows[shown.get(label, label)]["reuse_verified"],
                        tuple(manifest["arms"][shown.get(label, label)]["runs"])) for label in DECISION}

    reference = decision_rows(full)
    assert all(complete == 3 and verified for complete, verified, _ in reference.values())
    for labels, seeds in (([MLP_REMOVED], [(6, 3)]), (["Attention residual removed"], PAIRS)):
        subset = _launch(base, out, labels, seeds)
        assert subset["paper_arms"]["seeds"] == [list(pair) for pair in PAIRS]
        assert decision_rows(subset) == reference
        assert _verdict(subset)["verdict"] == "stands"
        rows = subset["paper_arms"]["rows"]
        assert rows[pa.DECISION_FROZEN]["status"] == "not selected"
        assert rows[pa.DECISION_FROZEN]["resolved_status"] == "ready"
        assert rows[labels[0]]["status"] == "ready" and rows[labels[0]]["complete"] == len(seeds)
    assert len(trained) == 11 + 1 + 3


def test_the_manifest_verifies_unselected_arms_against_their_resolved_config(tmp_path: Path) -> None:
    base = _base(tmp_path)
    plans = [_plan(base, pa.EVALUATED, seeds=[(23, 23)])]
    exp = pa.experiment_dir(tmp_path / "out", "tiny", "base_protocol")
    mlp = _plan(base, MLP_REMOVED, seeds=[(23, 23)])
    good = _fake_run(exp / "no_mlp", "3.40_no_mlp_s23_d23", mlp.cfg, seed=23, data_seed=23)
    other = pa.rebuild(mlp.cfg, {"m_p_mu_lr": float(mlp.cfg.m_p_mu_lr) * 2}, "test")
    stale = _fake_run(exp / "no_residual", "3.41_no_residual_s23_d23", other, seed=23, data_seed=23)
    manifest = _quiet(pa.build_manifest, exp, plans, base=base, budget="base_protocol", seeds=[(23, 23)])
    rows = manifest["paper_arms"]["rows"]
    assert manifest["arms"][MLP_REMOVED]["runs"] == [str(good)]
    assert rows[MLP_REMOVED]["status"] == "not selected" and rows[MLP_REMOVED]["reuse_verified"] is True
    # A run of another configuration filed under an unselected arm is reported, not listed unchecked.
    assert manifest["arms"]["Attention residual removed"]["runs"] == []
    assert [entry["run"] for entry in rows["Attention residual removed"]["stale"]] == [str(stale)]


def test_a_manifest_of_another_base_or_an_unreadable_one_contributes_no_seed_pairs(tmp_path: Path) -> None:
    base = _base(tmp_path)
    exp = pa.experiment_dir(tmp_path / "out", "tiny", "base_protocol")
    assert pa.manifest_seed_pairs(exp, base=base, budget="base_protocol", seeds=[(1, 1)]) == [(1, 1)]
    exp.mkdir(parents=True)
    (exp / "m2_manifest.json").write_text("{not json")
    assert pa.manifest_seed_pairs(exp, base=base, budget="base_protocol", seeds=[(1, 1)]) == [(1, 1)]
    record = {"paper_arms": {"base": "tiny", "budget": "base_protocol", "seeds": [[6, 3], [1, 1]]}}
    (exp / "m2_manifest.json").write_text(json.dumps(record))
    assert pa.manifest_seed_pairs(exp, base=base, budget="base_protocol", seeds=[(1, 1), (2, 2)]) == [
        (6, 3), (1, 1), (2, 2)]
    record["paper_arms"]["base"] = "another"
    (exp / "m2_manifest.json").write_text(json.dumps(record))
    assert pa.manifest_seed_pairs(exp, base=base, budget="base_protocol", seeds=[(1, 1)]) == [(1, 1)]


# ---- paper-arms-harness-2: an arm whose every change is inert ------------------------------------

def test_an_arm_whose_every_change_is_inert_is_inapplicable_and_rows_carry_their_notes(tmp_path: Path) -> None:
    passthrough = _base(tmp_path, "passthrough")                   # the live route's passthrough covariance
    assert passthrough.cfg.predictive_covariance == "passthrough"
    inert = _plan(passthrough, BLOCK_LEARNED)
    assert inert.status == "inapplicable" and "inert" in inert.reason and "family" in inert.reason
    assert inert.diff == {"family": ("gaussian_diagonal", "gaussian_block")}
    moment = _base(tmp_path, "moment", predictive_covariance="moment_matched")
    assert _plan(moment, BLOCK_LEARNED).status == "ready"         # moment matching makes the family live
    plans = _quiet(pa.build_plan, passthrough, budget="base_protocol", seeds=[(6, 3)], counts=False,
                   baseline_overrides={"lr": 3e-3})
    for plan in plans:
        if plan.kind == "vfe" and plan.status == "ready" and plan.diff:
            assert not all(key in plan.cfg.inactive_fields() for key in plan.diff), plan.label
    manifest = _quiet(pa.build_manifest, tmp_path / "exp", plans, base=passthrough, budget="base_protocol",
                      seeds=[(6, 3)])
    rows = manifest["paper_arms"]["rows"]
    assert any(plan.notes for plan in plans)
    for plan in plans:
        shown = pa.evaluated_label(passthrough.cfg) if plan.label == pa.EVALUATED else plan.label
        assert rows[shown]["notes"] == plan.notes, plan.label
    assert manifest["arms"][BLOCK_LEARNED]["inapplicable"] == inert.reason


# ---- paper-arms-harness-5: a base that already is an arm's configuration -------------------------

def test_a_base_that_already_is_an_arms_configuration_resolves_it_inapplicable(
        tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    cases = ((REFERENCE_BLOCK, dict(block_coordinates="reference")),
             (IDENTITY_NO_DECAY, dict(gauge_transport="off", sigma_weight_decay=0.0)))
    for label, over in cases:
        name = label.split()[0].lower()
        base = _base(tmp_path, name, **over)
        plan = _plan(base, label)
        assert plan.status == "inapplicable" and "already is this row's configuration" in plan.reason, label
        config = {**pa.CONFIG, "bases": [name], "arms": None, "seeds": [(6, 6)], "dry_run": True,
                  "out_root": str(tmp_path / "out")}
        _quiet(pa.main, config, bases={name: str(base.run_dir)})
        assert label[:56] in capsys.readouterr().out
    # An undeclared difference is still a defect of ARMS.
    with pytest.raises(pa.ArmDefinitionError, match="undeclared"):
        _quiet(pa.build_plan, _base(tmp_path, "gauge_on"), budget="base_protocol", seeds=[(6, 3)], counts=False,
               arms={"bad": {"group": "g", "slug": "bad", "vfe": {"gauge_transport": "off"}}})


# ---- paper-arms-harness-7 and gap-paper-arm-offlive-routes-6: the Cholesky table on a Fisher base -

def test_frame_free_table_and_covariant_stack_share_the_adamw_table_step_on_a_fisher_base(tmp_path: Path) -> None:
    fisher = _base(tmp_path, "fisher", **FISHER)
    chol, ref = _plan(fisher, BLOCK_CHOLESKY, counts=True), _plan(fisher, BLOCK_REFERENCE, counts=True)
    for plan in (chol, ref):
        assert plan.status == "ready", (plan.label, plan.reason)
        assert plan.diff["m_table_update_mode"] == ("fisher", "adamw"), plan.label
        assert plan.cfg.m_table_update_mode == "adamw" and plan.params["total"] > 0
        assert sum(d.startswith("table optimizer:") for d in plan.deviations) == 1, plan.label
    # The two rows differ in the frame and table-layout fields alone (gap-paper-arm-offlive-routes-6).
    assert set(pa.config_difference(chol.cfg, ref.cfg)) <= {
        "gauge_transport", "phi_scale", "m_phi_lr", "pos_phi", "prior_covariance_table", "block_coordinates",
        "decode_coordinates"}
    rows = _quiet(pa.build_manifest, tmp_path / "exp", [chol, ref], base=fisher, budget="base_protocol",
                  seeds=[(6, 3)])["paper_arms"]["rows"]
    for label in (BLOCK_CHOLESKY, BLOCK_REFERENCE):
        assert rows[label]["changes"]["m_table_update_mode"] == ["'fisher'", "'adamw'"]
        assert any(d.startswith("table optimizer:") for d in rows[label]["declared_deviations"])
    # An AdamW-table base needs no companion; neither row records one.
    adamw = _base(tmp_path, "adamw", pos_phi="none")
    for label in (BLOCK_CHOLESKY, BLOCK_REFERENCE):
        plan = _plan(adamw, label)
        assert plan.status == "ready" and "m_table_update_mode" not in plan.diff
        assert not any(d.startswith("table optimizer:") for d in plan.deviations)
    # Where the frame-free arm does not apply (positional frames), the covariant stack keeps the Fisher step.
    positional = _base(tmp_path, "positional", **{**FISHER, "pos_phi": "learned"})
    assert _plan(positional, BLOCK_CHOLESKY).status == "inapplicable"
    stack = _plan(positional, BLOCK_REFERENCE)
    assert stack.status == "ready" and "m_table_update_mode" not in stack.diff


def test_rest_and_all_arms_launches_plan_without_invalid_rows_on_a_fisher_base(tmp_path: Path, caches: Path) -> None:
    fisher = _base(tmp_path, "fisher", **FISHER)
    rest = [label for label in pa.ARMS if label not in (*pa.TIER_A, *pa.TIER_B)]
    plans = _quiet(pa.build_plan, fisher, budget="base_protocol", seeds=[(6, 3)], labels=rest, counts=False,
                   baseline_overrides={"lr": 3e-3})
    assert [plan.label for plan in plans if plan.status == "invalid"] == []
    config = {**pa.CONFIG, "bases": ["fisher"], "arms": None, "seeds": [(6, 3)], "dry_run": True,
              "out_root": str(tmp_path / "out")}
    assert _quiet(pa.main, config, bases={"fisher": str(fisher.run_dir)}) == 0


# ---- gap-paper-arm-offlive-routes-2: positional frames -------------------------------------------

def test_cholesky_arm_is_inapplicable_with_positional_frames_and_identity_rows_declare_their_loss(
        tmp_path: Path) -> None:
    learned = _base(tmp_path, "learned", pos_phi="learned", beta_attention_prior="causal", rope_mode="none")
    chol = _plan(learned, BLOCK_CHOLESKY)
    assert chol.status == "inapplicable" and "pos_phi='learned'" in chol.reason
    assert _plan(_base(tmp_path, "none", pos_phi="none"), BLOCK_CHOLESKY).status == "ready"
    for label in (IDENTITY_DECAYED, IDENTITY_NO_DECAY, pa.DECISION_CAPACITY):
        plan = _plan(learned, label)
        assert plan.status == "ready" and plan.diff["pos_phi"] == ("learned", "none"), label
        removed = [d for d in plan.deviations if d.startswith("positional frames removed")]
        assert len(removed) == 1 and "only positional channel" in removed[0], label
        assert f"declared difference: {removed[0]}" in plan.notes
        row = _quiet(pa.build_manifest, tmp_path / f"exp_{plan.slug}", [plan], base=learned,
                     budget="base_protocol", seeds=[(6, 3)])["paper_arms"]["rows"][label]
        assert removed[0] in row["declared_deviations"]
    # With ALiBi the positional frames are not the only channel; without positional frames nothing is declared.
    alibi = _base(tmp_path, "alibi", pos_phi="learned", beta_attention_prior="causal_alibi", rope_mode="none")
    (kept,) = [d for d in _plan(alibi, IDENTITY_NO_DECAY).deviations if d.startswith("positional frames removed")]
    assert "beta_attention_prior='causal_alibi'" in kept and "only positional channel" not in kept
    plain = _base(tmp_path, "plain", pos_phi="none")
    assert not any(d.startswith("positional frames") for d in _plan(plain, IDENTITY_NO_DECAY).deviations)


# ---- gap-paper-arm-offlive-routes-9: the intertwiner row's mean scale ----------------------------

def test_the_intertwiner_row_declares_its_mean_scale_change(tmp_path: Path) -> None:
    base = _base(tmp_path, "layernorm", sigma_init=15.0)
    plan = _plan(base, INTERTWINER)
    assert plan.status == "ready" and plan.cfg.norm_type_block == "mahalanobis"
    (scale,) = [d for d in plan.deviations if d.startswith("mean scale:")]
    assert "sigma_init=15" in scale and f"declared difference: {scale}" in plan.notes
    row = _quiet(pa.build_manifest, tmp_path / "exp", [plan], base=base, budget="base_protocol",
                 seeds=[(6, 3)])["paper_arms"]["rows"][INTERTWINER]
    assert scale in row["declared_deviations"]
    no_norm = _base(tmp_path, "no_norm", norm_type_block="none", layernorm_affine=False, sigma_init=15.0)
    kept = _plan(no_norm, INTERTWINER)
    assert kept.status == "ready" and not any(d.startswith("mean scale:") for d in kept.deviations)
    assert "(and final)" not in pa.__doc__                          # paper-arms-harness-13


# ---- gap-mixed-precision-routes-2: effective precision under fp16 --------------------------------

def test_effective_precision_follows_the_arithmetic_under_fp16() -> None:
    from vfe3.baseline import BaselineConfig

    islanded = _quiet(_tiny_cfg, amp_dtype="fp16")                  # linear decode without decode_logits_amp
    assert pa.effective_precision(islanded) == "float32"
    live = _quiet(_tiny_cfg, amp_dtype="fp16", decode_logits_amp=True)
    assert pa.effective_precision(live) == "fp16"
    assert pa.effective_precision(_quiet(_tiny_cfg, amp_dtype="bf16")) == "float32"
    assert pa.effective_precision(_quiet(_tiny_cfg, amp_dtype="bf16", decode_logits_amp=True)) == "bf16"
    baseline = BaselineConfig.matched_to(islanded)
    assert baseline.amp_dtype == "fp16" and pa.effective_precision(baseline) == "fp16"
    with pytest.raises(pa.ArmDefinitionError, match="amp_dtype"):
        pa.check_baseline_arm(islanded, baseline)
    assert pa.check_baseline_arm(islanded, BaselineConfig.matched_to(islanded, amp_dtype=None)) == {}
    assert pa.check_baseline_arm(live, BaselineConfig.matched_to(live)) == {}


# ---- paper-arms-harness-8: a base recorded with trust_resume_checkpoint=True ---------------------

def test_a_base_recorded_with_trust_resume_checkpoint_is_reused_beside_runs_trained_now(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vfe3.paper.runs import resolve_manifest

    base = _base(tmp_path, resume_from="old/checkpoints/step_4.pt", trust_resume_checkpoint=True)
    seeds = [(6, 3), (23, 23)]
    plan = _plan(base, pa.EVALUATED, seeds=seeds)
    assert plan.cfg.trust_resume_checkpoint is True and plan.reused == [(6, 3)]
    trained = _fake_trainers(monkeypatch)
    path = _quiet(pa.run_plan, base, [plan], budget="base_protocol", out_root=tmp_path / "out", device="cpu",
                  logger=LOGGER)
    assert trained == ["evaluated_s23_d23"]
    exp = pa.experiment_dir(tmp_path / "out", "tiny", "base_protocol")
    stale: List[Dict[str, str]] = []
    run = pa.completed_run(exp, pa.EVALUATED, 23, 23, base=base, budget="base_protocol", cfg=plan.cfg,
                           dataset=plan.dataset, stale=stale)
    assert run is not None and stale == []
    manifest = json.loads(path.read_text(encoding="utf-8"))
    shown = pa.evaluated_label(base.cfg)
    assert manifest["arms"][shown]["runs"] == [str(base.run_dir), str(run)]
    arm = next(a for a in _quiet(resolve_manifest, manifest).arms if a.label == shown)
    assert arm.homogeneity["homogeneous"], arm.homogeneity["violations"]
    # A relaunch is not refused by the run trained after the base.
    _quiet(pa.run_plan, base, [plan], budget="base_protocol", out_root=tmp_path / "out", device="cpu", logger=LOGGER)
    assert trained == ["evaluated_s23_d23"]


# ---- paper-arms-harness-9: the base's recorded training-data identity ----------------------------

def _cache_identity(cache_dir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"tokenizer_tag": "tiktoken", "max_tokens": None}
    for key, split in (("train", "train"), ("val", "validation"), ("test", "test")):
        tokens = torch.load(datasets_mod.cache_path(BPE, split, cache_dir=cache_dir), weights_only=True)
        out[f"{key}_data_sha256"] = hashlib.sha256(tokens.to(torch.long).numpy().tobytes()).hexdigest()
        out[f"{key}_data_n_tokens"] = int(tokens.numel())
    return out


def test_base_reuse_compares_the_recorded_training_data_identity(tmp_path: Path, caches: Path) -> None:
    current = _cache_identity(caches)
    assert pa.data_identity_mismatch(current, BPE) is None
    cases = {"capped": ({**current, "max_tokens": 1000}, "max_tokens"),
             "rebuilt": ({**current, "train_data_sha256": "0" * 64}, "train_data_sha256"),
             "shorter": ({**current, "test_data_n_tokens": 7}, "test_data_n_tokens")}
    for name, (provenance, field) in cases.items():
        base = _base(tmp_path, name, provenance=provenance)
        plan = _plan(base, pa.EVALUATED, seeds=[(6, 3), (23, 23)])
        assert plan.reused == [], name
        assert any("not reused" in note and field in note for note in plan.notes), (name, plan.notes)
        stale: List[Dict[str, str]] = []
        exp = pa.experiment_dir(tmp_path / f"out_{name}", name, "base_protocol")
        assert pa.completed_run(exp, pa.EVALUATED, 6, 3, base=base, budget="base_protocol", cfg=plan.cfg,
                                dataset=plan.dataset, stale=stale) is None
        assert [entry["run"] for entry in stale] == [str(base.run_dir)] and field in stale[0]["reason"]
    for name, provenance in (("matching", current), ("unrecorded", {})):
        base = _base(tmp_path, name, provenance=provenance)
        assert _plan(base, pa.EVALUATED, seeds=[(6, 3), (23, 23)]).reused == [(6, 3)], name


def test_the_data_identity_helper_matches_what_a_run_records(tmp_path: Path, caches: Path) -> None:
    from vfe3.experiment import run_vfe

    cfg = _quiet(_tiny_cfg, max_steps=2, eval_interval=0, checkpoint_interval=0)
    run = _quiet(run_vfe, cfg, dataset=BPE, out_root=tmp_path / "runs", seed=1, data_seed=1, label="identity",
                 device="cpu", logger=LOGGER)
    provenance = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["train_data_sha256"] is not None
    assert pa.data_identity_mismatch(provenance, BPE) is None
    assert {key: provenance[key] for key in _cache_identity(caches)} == _cache_identity(caches)


# ---- paper-arms-harness-11: word-level rows under one budget directory ---------------------------

WORD_ARMS = ["Gaussian attention", "Conventional transformer, matched"]


def test_word_level_rows_are_trained_once_whichever_budget_launches_them(
        tmp_path: Path, caches: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    trained = _fake_trainers(monkeypatch)
    base = _base(tmp_path)
    out = tmp_path / "out"
    first = _launch(base, out, WORD_ARMS, [(1, 1)], budget="base_protocol")
    assert sorted(trained) == ["word_gaussian_s1_d1", "word_tf_matched_s1_d1"]
    word_dir = pa.experiment_dir(out, "tiny", pa.WORD_LEVEL_BUDGET)
    for label in WORD_ARMS:
        (run,) = first["arms"][label]["runs"]
        assert Path(run).parent == word_dir / pa.ARMS[label]["slug"], label
    second = _launch(base, out, WORD_ARMS, [(1, 1)], budget="one_epoch")
    assert len(trained) == 2
    assert {label: second["arms"][label]["runs"] for label in WORD_ARMS} == {
        label: first["arms"][label]["runs"] for label in WORD_ARMS}
    # A word run an earlier launch trained in the launch budget's own directory is still found there.
    legacy_out = tmp_path / "legacy"
    plan = _plan(base, "Gaussian attention", seeds=[(1, 1)])
    legacy = _write_run(pa.experiment_dir(legacy_out, "tiny", "base_protocol") / "word_gaussian"
                        / "3.20_word_gaussian_s1_d1", serialize_config(experiment._with_seed(plan.cfg, 1)), WORD, 1,
                        1, 3.1)
    third = _launch(base, legacy_out, ["Gaussian attention"], [(1, 1)], budget="base_protocol")
    assert third["arms"]["Gaussian attention"]["runs"] == [str(legacy)] and len(trained) == 2


# ---- paper-arms-harness-12: baseline rows' defining choices --------------------------------------

def test_baseline_rows_record_their_defining_choices_in_the_manifest_changes(tmp_path: Path) -> None:
    from vfe3.baseline import BaselineConfig

    import train_baseline  # noqa: F401
    base = _base(tmp_path, beta_attention_prior="causal")
    reference = BaselineConfig.matched_to(pa.resolved_base(base, "base_protocol", base.dataset), lr=3e-3)
    expected = {
        "Per-type read table, ALiBi (as run)": {"per_type_table": ("none", "block_read")},
        "Per-type read table, ALiBi, realization optimizer": {
            "per_type_table": ("none", "block_read"), "adam_beta2": (reference.adam_beta2, 0.999),
            "weight_decay": (reference.weight_decay, 0.0),
            "per_type_table_weight_decay": (reference.per_type_table_weight_decay, 0.1)},
        "Per-type read-and-write table, ALiBi": {"per_type_table": ("none", "block_rw")},
        "Per-type read table, ALiBi, shared post-norm": {
            "per_type_table": ("none", "block_read"), "norm_placement": (reference.norm_placement, "post_shared")},
        "Per-type table of the frame table's size": {"per_type_table": ("none", "block_read")},
        "Matched context-processing parameters": {}}
    plans = [_plan(base, label) for label in expected]
    for plan in plans:
        want = {key: value for key, value in expected[plan.label].items() if value[0] != value[1]}
        assert plan.defining == want, plan.label
        assert not set(plan.defining) & set(plan.diff), plan.label         # protocol differences stay in diff
    assert plans[0].diff == {"beta_attention_prior": ("causal", "causal_alibi")}
    manifest = _quiet(pa.build_manifest, tmp_path / "exp", plans, base=base, budget="base_protocol", seeds=[(6, 3)])
    for plan in plans:
        changes = manifest["paper_arms"]["rows"][plan.label]["changes"]
        assert changes == {key: [repr(a), repr(b)] for key, (a, b) in {**plan.defining, **plan.diff}.items()}
    assert "per_type_table: 'none'->'block_rw'" in pa.format_plan(base, plans, budget="base_protocol")


# ---- paper-arms-harness-10: an interrupted run resumes from its last checkpoint ------------------

class _Interrupted(BaseException):
    r"""Stands in for Ctrl+C: not an ``Exception``, so ``run_vfe`` records no failure and the run stays unfinished."""


def _final_state(run: Path) -> Dict[str, torch.Tensor]:
    return torch.load(Path(run) / "final_model.pt", weights_only=True)["model_state"]


def _metrics(run: Path) -> List[Dict[str, str]]:
    import csv

    with (Path(run) / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_the_resumable_checkpoint_cadence_leaves_the_trained_weights_unchanged(tmp_path: Path, caches: Path) -> None:
    from vfe3.experiment import run_vfe

    cfg = _quiet(_tiny_cfg, max_steps=4, eval_interval=2, log_interval=1, checkpoint_interval=4)
    plain = _quiet(run_vfe, cfg, dataset=BPE, out_root=tmp_path / "plain", seed=1, data_seed=1, label="plain_s1_d1",
                   device="cpu", logger=LOGGER)
    every = _quiet(run_vfe, pa.resumable_cadence(cfg, 2), dataset=BPE, out_root=tmp_path / "every", seed=1,
                   data_seed=1, label="every_s1_d1", device="cpu", logger=LOGGER)
    a, b = _final_state(plain), _final_state(every)
    assert a.keys() == b.keys() and all(torch.equal(a[key], b[key]) for key in a)
    assert sorted(path.name for path in (every / "checkpoints").glob("step_*.pt")) == ["step_4.pt"]
    assert pa.reuse_mismatch(every, cfg, dataset=BPE, seed=1, data_seed=1) is None     # one arm at either cadence
    with pytest.raises(ValueError, match="checkpoint_every"):
        pa.resumable_cadence(cfg, 0)


def test_an_interrupted_run_resumes_from_its_last_checkpoint_and_matches_an_uninterrupted_one(
        tmp_path: Path, caches: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vfe3.run_artifacts as ra
    from vfe3.experiment import run_vfe

    base = _base(tmp_path, max_steps=4, eval_interval=2, log_interval=1, checkpoint_interval=4)
    plan = _plan(base, MLP_REMOVED, seeds=[(1, 1)])
    out = tmp_path / "out"
    exp = pa.experiment_dir(out, "tiny", "base_protocol")
    log_metrics = ra.RunArtifacts.log_metrics

    def interrupt_after_step_two(self: Any, row: Any) -> None:
        log_metrics(self, row)
        if float(row["step"]) == 2:                    # between steps, after step 2's checkpoint and row
            raise _Interrupted()

    with monkeypatch.context() as patch:
        patch.setattr(ra.RunArtifacts, "log_metrics", interrupt_after_step_two)
        with pytest.raises(_Interrupted):
            _quiet(pa.run_plan, base, [plan], budget="base_protocol", out_root=out, device="cpu", logger=LOGGER,
                   checkpoint_every=2)
    (unfinished,) = list((exp / "no_mlp").iterdir())
    assert not (unfinished / "summary.json").exists() and pa.failed_run_reason(unfinished) is None
    checkpoint = pa.resumable_checkpoint(exp, MLP_REMOVED, 1, 1, cfg=plan.cfg, dataset=plan.dataset)
    assert checkpoint is not None and checkpoint.parent.parent == unfinished and checkpoint.name == "step_2.pt"

    path = _quiet(pa.run_plan, base, [plan], budget="base_protocol", out_root=out, device="cpu", logger=LOGGER,
                  checkpoint_every=2)
    (run,) = [Path(entry) for entry in json.loads(path.read_text(encoding="utf-8"))["arms"][MLP_REMOVED]["runs"]]
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["resume_source"]["saved_step"] == 2 and run != unfinished
    reference = _quiet(run_vfe, plan.cfg, dataset=BPE, out_root=tmp_path / "reference", seed=1, data_seed=1,
                       label="reference_s1_d1", device="cpu", logger=LOGGER)
    a, b = _final_state(reference), _final_state(run)
    assert a.keys() == b.keys() and all(torch.equal(a[key], b[key]) for key in a)
    assert summary["test_ce"] == json.loads((reference / "summary.json").read_text(encoding="utf-8"))["test_ce"]
    # The resumed run's metrics.csv holds the interrupted run's rows up to the resume step, then its own.
    columns = ("step", "train_loss", "lr_mu", "val_ce")
    assert [[row[c] for c in columns] for row in _metrics(run)] == [
        [row[c] for c in columns] for row in _metrics(reference)]
    assert pa.stitch_resumed_metrics(run) == 0                                       # once only
    # A relaunch finds the pair complete.
    trained = _fake_trainers(monkeypatch)
    _quiet(pa.run_plan, base, [plan], budget="base_protocol", out_root=out, device="cpu", logger=LOGGER,
           checkpoint_every=2)
    assert trained == []


def test_only_a_launch_with_checkpoint_every_resumes_and_only_a_run_of_the_plans_config(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = _base(tmp_path)
    plan = _plan(base, MLP_REMOVED, seeds=[(1, 1)])
    exp = pa.experiment_dir(tmp_path / "out", "tiny", "base_protocol")
    interrupted = _fake_run(exp / "no_mlp", "20261031-101010_no_mlp_s1_d1", plan.cfg, test_ce=None)
    (interrupted / "checkpoints").mkdir()
    for step in (2, 3):
        (interrupted / "checkpoints" / f"step_{step}.pt").write_bytes(b"")
    other = _fake_run(exp / "no_mlp", "20261031-111111_no_mlp_s1_d1",
                      pa.rebuild(plan.cfg, {"m_p_mu_lr": float(plan.cfg.m_p_mu_lr) * 2}, "test"), test_ce=None)
    (other / "checkpoints").mkdir()
    (other / "checkpoints" / "step_3.pt").write_bytes(b"")
    calls: List[Dict[str, Any]] = []

    def run_vfe(cfg: Any, **kwargs: Any) -> Path:
        calls.append({"checkpoint_interval": cfg.checkpoint_interval, **kwargs})
        raise _Interrupted()

    monkeypatch.setattr(experiment, "run_vfe", run_vfe)
    for every, expected in ((None, None), (3, interrupted / "checkpoints" / "step_3.pt")):
        with pytest.raises(_Interrupted):
            _quiet(pa.run_plan, base, [plan], budget="base_protocol", out_root=tmp_path / "out", device="cpu",
                   logger=LOGGER, checkpoint_every=every)
        assert calls[-1].get("resume_from") == expected
        assert calls[-1]["checkpoint_interval"] == (plan.cfg.checkpoint_interval if every is None else every)
    with pytest.raises(SystemExit, match="checkpoint_every"):
        _quiet(pa.main, {**pa.CONFIG, "bases": ["tiny"], "checkpoint_every": 0, "dry_run": True},
               bases={"tiny": str(base.run_dir)})


# ---- paper-arms-harness-14: the plan totals ------------------------------------------------------

def test_plan_totals_leave_out_reused_pairs_and_report_the_matched_baseline_apart(tmp_path: Path) -> None:
    import train_baseline  # noqa: F401
    base = _base(tmp_path)
    plans = _quiet(pa.build_plan, base, budget="base_protocol", seeds=[(6, 3), (23, 23)],
                   labels=[pa.EVALUATED, COMPUTE], baseline_overrides={"lr": 3e-3})
    evaluated, matched = plans
    assert evaluated.reused == [(6, 3)] and matched.status == "deferred"
    assert matched.est_hours == evaluated.est_hours is not None
    totals = pa.plan_totals(plans)
    assert totals["vfe_hours"] == pytest.approx(evaluated.est_hours * 1)
    assert totals["compute_matched_hours"] == pytest.approx(matched.est_hours * 2)
    disk = {plan.label: pa._plan_storage(plan)[1] for plan in plans}
    assert all(value is not None and value > 0 for value in disk.values())
    assert totals["disk_gb"] == pytest.approx(disk[pa.EVALUATED] * 1 + disk[COMPUTE] * 2)
    printed = pa.format_plan(base, plans, budget="base_protocol")
    assert "compute-matched baseline" in printed and "reused" in printed
