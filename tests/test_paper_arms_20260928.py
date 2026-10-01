r"""Main-results arms (``paper_arms.py``) and the library training run (:func:`vfe3.experiment.run_vfe`).

Everything runs on tiny synthetic bases (K = 4, two heads, two layers, vocabulary 7) and tiny token
caches written by the test into a temporary cache directory. Covered: the arm labels against the
manuscript rows, base loading (legacy fields dropped, the working-tree default flips pinned), arm
resolution with the difference assertion, inapplicable and unavailable arms, the one-epoch budget,
skip-completed, the manifest read back by :mod:`vfe3.paper`, the dry run, an end-to-end launch of
two paired seeds, and ``run_vfe`` reproducing ``train_vfe3._run_once`` bitwise.
"""

from __future__ import annotations

from tests.profiles import test_config

import ast
import dataclasses
import json
import logging
import math
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import pytest
import torch

import paper_arms as pa
import vfe3.data.datasets as datasets_mod
import vfe3.run_artifacts as ra
import vfe3.train as T
from vfe3.config import VFE3Config, serialize_config
from vfe3.data.word_vocab import WORD_EOS, WORD_UNK, WordVocabulary, vocabulary_path

ROOT = Path(__file__).resolve().parents[1]
BPE = "wikitext-103"
WORD = pa.WORD_DATASET
V_BPE, V_WORD = 7, 9
N_TOKENS = {"train": 90, "validation": 30, "test": 30}
PINNED = {"norm_type_block": "layernorm", "layernorm_affine": True, "skip_belief_sigma_update": True}


@pytest.fixture(autouse=True)
def _cpu_one_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.set_num_threads(1)
    monkeypatch.setattr(ra, "_run_figures_isolated", lambda *args, **kwargs: True)
    monkeypatch.setattr(T, "_save_eval_attention_maps", lambda *args, **kwargs: None)


def _tiny_cfg(**over: Any) -> VFE3Config:
    values: Dict[str, Any] = dict(
        vocab_size=V_BPE, embed_dim=4, n_heads=2, batch_size=2, max_seq_len=4, eval_stride=2, n_layers=2,
        max_steps=4, warmup_steps=0, log_interval=0, eval_interval=2, checkpoint_interval=4, e_phi_lr=0.0,
        family="gaussian_diagonal", gauge_group="block_glk", attention_mode="projected_predictive",
        attention_residual=True, decode_mode="linear", use_block_mlp=True, block_mlp_expansion=2,
        generate_embedding_figures=False, sigma_weight_decay=0.2, pos_phi="learned", seed=0,
        relational_input_scope="full", relational_output_scope="full", **PINNED)
    values.update(over)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return test_config(**values)


@pytest.fixture()
def caches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    r"""Tiny ``wikitext-103`` (BPE-tagged) and ``wikitext-103-word-eos`` caches in a temporary directory."""
    root = tmp_path / "cache"
    root.mkdir()
    generator = torch.Generator().manual_seed(0)
    WordVocabulary([WORD_UNK, WORD_EOS, *"abcdefg"[:V_WORD - 2]], eos=True).save(
        vocabulary_path(WORD, tag="word", cache_dir=root))
    for split, n in N_TOKENS.items():
        torch.save(torch.randint(0, V_BPE, (n,), generator=generator), datasets_mod.cache_path(BPE, split, cache_dir=root))
        path = datasets_mod.cache_path(WORD, split, cache_dir=root)
        torch.save(torch.randint(0, V_WORD, (n,), generator=generator), path)
        Path(str(path) + ".meta.json").write_text(json.dumps(
            {"n_tokens": n, "n_chars_source": 5 * n, "vocab_size": V_WORD, "min_count": 0, "eos": True}))
    monkeypatch.setattr(datasets_mod, "default_cache_dir", lambda: root)
    return root


def _write_base(
    root:    Path,
    cfg:     VFE3Config,

    *,
    dataset: str = BPE,
    extra:   Optional[Dict[str, Any]] = None,
    drop:    tuple = (),
) -> Path:
    run = root / "base_run"
    run.mkdir(parents=True)
    config = {key: value for key, value in serialize_config(cfg).items() if key not in drop}
    config.update(extra or {})
    (run / "config.json").write_text(json.dumps({"config": config, "dataset": dataset, "n_params": 123}))
    (run / "summary.json").write_text(json.dumps({"n_steps": 100, "wall_time_s": 20.0}))
    return run


def _plans(base: pa.Base, **kwargs: Any) -> Dict[str, pa.ArmPlan]:
    plans = pa.build_plan(base, budget="one_epoch", seeds=[(1, 1), (2, 5)], **kwargs)
    return {plan.label: plan for plan in plans}


# ---- labels ------------------------------------------------------------------------------------

def _paper_artifacts_rows() -> Dict[str, str]:
    r"""``{label: group constant name}`` of paper_artifacts.CONFIG["arms"], read without importing it."""
    tree = ast.parse((ROOT / "paper_artifacts.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "CONFIG" for t in node.targets):
            config = node.value
            arms = next(v for k, v in zip(config.keys, config.values) if k.value == "arms")
            rows = {}
            for key, value in zip(arms.keys, arms.values):
                group = next(v for k, v in zip(value.keys, value.values) if k.value == "group")
                rows[key.value] = group.id
            return rows
    raise AssertionError("paper_artifacts.CONFIG not found")


def test_arm_labels_and_groups_are_the_manuscript_rows() -> None:
    rows = _paper_artifacts_rows()
    assert list(pa.ARMS) == list(rows)
    groups = {"_RETRAINING": pa._RETRAINING, "_BASELINES": pa._BASELINES, "_WORD_LEVEL": pa._WORD_LEVEL}
    assert [spec["group"] for spec in pa.ARMS.values()] == [groups[rows[label]] for label in rows]
    assert len({spec["slug"] for spec in pa.ARMS.values()}) == len(pa.ARMS)


# ---- bases and budgets -------------------------------------------------------------------------

def test_load_base_pins_wip_flipped_fields_and_records_legacy_fields(tmp_path: Path) -> None:
    run = _write_base(tmp_path, _tiny_cfg(), extra={"mean_only": False}, drop=tuple(PINNED))
    base = pa.load_base("tiny", run)
    assert base.pinned == pa.WIP_FLIPPED_DEFAULTS == PINNED
    assert {key: getattr(base.cfg, key) for key in PINNED} == PINNED
    assert base.dropped == {"mean_only": False}
    assert not set(PINNED) & set(base.defaulted)
    assert base.tokens_per_s == pytest.approx(100 * 2 * 4 / 20.0)
    # A recorded value wins over the pre-flip default.
    other = tmp_path / "other"
    recorded = pa.load_base("tiny", _write_base(other, _tiny_cfg(norm_type_block="none", layernorm_affine=False)))
    assert recorded.pinned["norm_type_block"] == "none" and recorded.cfg.norm_type_block == "none"


def test_one_epoch_budget_gives_twenty_evaluations_and_one_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(datasets_mod, "cached_token_count", lambda dataset, split: 116_840_318)
    budget = pa.one_epoch(_tiny_cfg(batch_size=64, max_seq_len=128), BPE)
    assert budget == {"max_steps": 14_263, "eval_interval": 713, "checkpoint_interval": 14_263}
    assert budget["max_steps"] // budget["eval_interval"] == pa.M3_POINTS
    assert pa.one_epoch(_tiny_cfg(batch_size=32, max_seq_len=128), BPE)["max_steps"] == 28_526


def test_seed_pairs() -> None:
    assert pa.seed_pairs([6, (7, 3)]) == [(6, 6), (7, 3)]
    for bad in ([6, 6], [-1], [(1, 2, 3)], [True]):
        with pytest.raises(ValueError):
            pa.seed_pairs(bad)


# ---- arm resolution ----------------------------------------------------------------------------

def test_vfe_arms_differ_from_the_base_only_in_declared_fields(tmp_path: Path, caches: Path) -> None:
    from vfe3.model.model import VFEModel

    base = pa.load_base("tiny", _write_base(tmp_path, _tiny_cfg()))
    plans = _plans(base)
    known = {f.name for f in dataclasses.fields(VFE3Config)}
    for label, spec in pa.ARMS.items():
        plan = plans[label]
        if "baseline" in spec:
            continue
        if any(name not in known for name in spec.get("requires", ())):
            assert plan.status == "unavailable", label
            continue
        assert plan.status == "ready", (label, plan.reason)
        declared = set(spec["vfe"]) | set(spec.get("coerced", ()))
        assert set(plan.diff) <= declared, label
        assert plan.seeds == [(1, 1), (2, 5)]
    assert plans[pa.EVALUATED].diff == {}
    assert set(plans["Identity frames, decayed stored covariance"].diff) == {
        "gauge_transport", "phi_scale", "pos_phi", "m_phi_lr"}
    assert plans["Learned frames, stored covariance without decay"].diff == {"sigma_weight_decay": (0.2, 0.0)}
    word = plans["Gaussian attention"]
    assert word.dataset == WORD and set(word.diff) == {"vocab_size", "decode_ce_checkpoint"}
    assert word.cfg.vocab_size == V_WORD and word.cfg.decode_ce_checkpoint == "always"
    assert word.cfg.max_steps == math.ceil(N_TOKENS["train"] / (2 * 4))
    evaluated = plans[pa.EVALUATED].cfg
    assert (evaluated.max_steps, evaluated.eval_interval, evaluated.checkpoint_interval) == (12, 1, 12)
    if "decode_coordinates" in known:
        gate = plans["Intertwiner-constrained nonlinear block"].cfg
        assert (gate.block_mlp_mode, gate.norm_type_block, gate.decode_coordinates) == (
            "gauge_gate", "mahalanobis", "reference")
    # The two-vocabulary extrapolation equals a direct build at the true vocabulary.
    for label in (pa.EVALUATED, "Gaussian attention", "Identity frames, decayed stored covariance"):
        cfg = plans[label].cfg
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = VFEModel(cfg)
        assert plans[label].params["stored"] == sum(p.numel() for p in model.parameters()), label
        assert plans[label].params["trainable"] == sum(
            p.numel() for p in model.parameters() if p.requires_grad), label
    assert plans["Identity frames, decayed stored covariance"].params["trainable"] < plans[pa.EVALUATED].params["trainable"]


def test_difference_assertion_rejects_undeclared_or_rewritten_changes(tmp_path: Path, caches: Path) -> None:
    base = pa.load_base("tiny", _write_base(tmp_path, _tiny_cfg()))
    bad_arms = {"no coercions declared": {"group": "g", "slug": "bad", "vfe": {"gauge_transport": "off"}}}
    with pytest.raises(pa.ArmDefinitionError, match="undeclared.*phi_scale"):
        pa.build_plan(base, budget="one_epoch", seeds=[(1, 1)], arms=bad_arms, counts=False)
    base_cfg = pa.resolved_base(base, "one_epoch", BPE)
    arm_cfg = pa.rebuild(base_cfg, {"gauge_transport": "off", "phi_scale": 0.5}, "test")
    with pytest.raises(pa.ArmDefinitionError, match="did not survive"):
        pa.check_vfe_arm(base_cfg, arm_cfg, {"gauge_transport": "off", "phi_scale": 0.5}, pa.GAUGE_OFF_COERCED)
    same = pa.rebuild(base_cfg, {"sigma_weight_decay": 0.2}, "test")
    with pytest.raises(pa.ArmDefinitionError, match="base configuration itself"):
        pa.check_vfe_arm(base_cfg, same, {"sigma_weight_decay": 0.2})


def test_inapplicable_arms_on_a_base_without_residual_decay_or_layernorm(tmp_path: Path, caches: Path) -> None:
    cfg = _tiny_cfg(attention_residual=False, sigma_weight_decay=0.0, norm_type_block="none",
                    layernorm_affine=False)
    plans = _plans(pa.load_base("tiny", _write_base(tmp_path, cfg)), counts=False)
    for label in ("Learned frames, stored covariance without decay", "Identity frames, decayed stored covariance",
                  "Attention residual removed"):
        assert plans[label].status == "inapplicable", label
    identity = plans["Identity frames, stored covariance without decay"]
    assert identity.status == "ready" and "sigma_weight_decay" not in identity.diff
    if "decode_coordinates" in {f.name for f in dataclasses.fields(VFE3Config)}:
        gate = plans["Intertwiner-constrained nonlinear block"]
        assert gate.status == "ready" and gate.cfg.norm_type_block == "none"
        assert "norm_type_block" not in gate.diff
    # A word-level base has no word-level arm to add.
    word_base = pa.load_base("tiny", _write_base(tmp_path / "w", _tiny_cfg(vocab_size=V_WORD), dataset=WORD))
    word_plans = _plans(word_base, counts=False)
    assert word_plans["Gaussian attention"].status == "inapplicable"
    assert word_plans["Conventional transformer, matched"].status in ("inapplicable", "unavailable")


def test_unavailable_arms_when_code_is_missing(tmp_path: Path, caches: Path,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    base = pa.load_base("tiny", _write_base(tmp_path, _tiny_cfg()))
    arms = {"future arm": {"group": "g", "slug": "future", "vfe": {"no_such_field": 1},
                           "requires": ("no_such_field",)},
            "Matched context-processing parameters": pa.ARMS["Matched context-processing parameters"]}
    monkeypatch.setitem(sys.modules, "train_baseline", None)          # import raises ImportError
    plans = {p.label: p for p in pa.build_plan(base, budget="one_epoch", seeds=[(1, 1)], arms=arms)}
    assert plans["future arm"].status == "unavailable" and "no_such_field" in plans["future arm"].reason
    baseline = plans["Matched context-processing parameters"]
    assert baseline.status == "unavailable" and "baseline code missing" in baseline.reason


def test_baseline_arms_match_the_protocol_when_the_baseline_exists(tmp_path: Path, caches: Path) -> None:
    pytest.importorskip("vfe3.baseline")
    pytest.importorskip("train_baseline")
    base = pa.load_base("tiny", _write_base(tmp_path, _tiny_cfg()))
    plans = _plans(base)
    matched = plans["Matched context-processing parameters"]
    assert matched.status == "ready" and matched.diff == {}
    assert matched.cfg.model_family == "conventional_transformer"
    assert (matched.cfg.embed_dim, matched.cfg.max_steps, matched.cfg.eval_stride) == (4, 12, 2)
    assert plans["Matched measured compute"].status == "deferred"
    table = plans["Per-type table of the frame table's size"]
    assert table.cfg.per_type_table == "block_read"
    word = plans["Conventional transformer, matched"]
    assert word.cfg.vocab_size == V_WORD and word.cfg.decode_ce_checkpoint == "always"
    # The read map has the size of the VFE token-frame table.
    assert table.params["per_type_tables"] - matched.params["per_type_tables"] == V_BPE * 4 * 2
    rejected = dataclasses.replace(matched.cfg, eval_stride=4)
    with pytest.raises(pa.ArmDefinitionError, match="eval_stride"):
        pa.check_baseline_arm(matched.matched_cfg, rejected)


# ---- runs on disk, the manifest and the dry run ------------------------------------------------

def _fake_run(arm_dir: Path, name: str, cfg: VFE3Config, *, test_ce: Optional[float] = 3.0,
              seed: int = 1, data_seed: int = 1) -> Path:
    run = arm_dir / name
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"config": {**serialize_config(cfg), "seed": seed},
                                                 "dataset": BPE}))
    (run / "provenance.json").write_text(json.dumps({"seed": seed, "data_seed": data_seed}))
    if test_ce is not None:
        (run / "summary.json").write_text(json.dumps({
            "test_ce": test_ce, "test_best_ce": test_ce, "best_val_ppl": 20.0, "best_step": 4,
            "n_steps": 4, "wall_time_s": 1.0}))
    return run


def test_skip_completed_and_the_manifest_vfe3_paper_reads(tmp_path: Path, caches: Path) -> None:
    from vfe3.paper import tables as pt
    from vfe3.paper.runs import resolve_manifest

    base = pa.load_base("tiny", _write_base(tmp_path, _tiny_cfg()))
    seeds = [(1, 1), (2, 5)]
    plans = pa.build_plan(base, budget="one_epoch", seeds=seeds, counts=False)
    exp = pa.experiment_dir(tmp_path / "out", "tiny", "one_epoch")
    cfg = {plan.label: plan.cfg for plan in plans}
    evaluated = exp / "evaluated"
    complete = _fake_run(evaluated, "3.21_evaluated_s1_d1", cfg[pa.EVALUATED])
    _fake_run(evaluated, "20260928-101010_evaluated_s2_d5", cfg[pa.EVALUATED], test_ce=None, seed=2, data_seed=5)
    _fake_run(evaluated, "9.99_evaluated_s2_d5", cfg[pa.EVALUATED], test_ce=float("nan"), seed=2, data_seed=5)
    second = _fake_run(evaluated, "3.30_evaluated_s2_d5_2", cfg[pa.EVALUATED], seed=2, data_seed=5)
    _fake_run(evaluated, "3.40_evaluated_s12_d5", cfg[pa.EVALUATED], seed=12, data_seed=5)
    no_mlp = _fake_run(exp / "no_mlp", "3.50_no_mlp_s1_d1", cfg["MLP removed"])
    assert pa.find_complete_run(evaluated, pa.run_label("evaluated", 1, 1)) == complete
    assert pa.find_complete_run(evaluated, pa.run_label("evaluated", 2, 5)) == second
    assert pa.find_complete_run(evaluated, pa.run_label("evaluated", 3, 3)) is None
    assert pa.find_complete_run(exp / "absent", "x") is None

    manifest = pa.build_manifest(exp, plans, base=base, budget="one_epoch", seeds=seeds)
    assert list(manifest["arms"]) == list(pa.ARMS)
    assert manifest["arms"][pa.EVALUATED]["runs"] == [str(complete), str(second)]
    assert manifest["arms"]["MLP removed"]["runs"] == [str(no_mlp)]
    assert manifest["evaluated_run"] == str(complete) and manifest["selection"] == "best"
    path = pa.write_manifest(exp, manifest)
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    resolved = resolve_manifest(reloaded)
    assert resolved.problems == []
    assert [arm.label for arm in resolved.arms] == list(pa.ARMS)
    arm = resolved.arms[0]
    assert [run.run_dir for run in arm.runs] == [complete, second]
    assert arm.homogeneity["homogeneous"] and arm.homogeneity["data_seeds"] == [1, 5]
    assert resolved.evaluated is not None and resolved.evaluated.run_dir == complete
    _, markdown, rows = pt.main_results_table(resolved, read_checkpoints=False)
    assert pa.EVALUATED in markdown and len(rows) == len(pa.ARMS)


def test_dry_run_prints_the_plan_and_trains_nothing(tmp_path: Path, caches: Path, capsys,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    import vfe3.experiment as experiment

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a dry run must not train")

    monkeypatch.setattr(experiment, "run_vfe", refuse)
    run = _write_base(tmp_path, _tiny_cfg())
    out = tmp_path / "out"
    config = {**pa.CONFIG, "bases": ["tiny"], "seeds": [1, (2, 5)], "out_root": str(out), "dry_run": True}
    assert pa.main(config, bases={"tiny": str(run)}) == 0
    printed = capsys.readouterr().out
    for label in pa.ARMS:
        assert label[:56] in printed
    assert "[(1, 1), (2, 5)]" in printed and "changes: gauge_transport" in printed
    assert not out.exists()


# ---- training ----------------------------------------------------------------------------------

def test_launch_trains_paired_seeds_skips_completed_and_writes_the_manifest(
        tmp_path: Path, caches: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vfe3.experiment as experiment
    from vfe3.paper.runs import resolve_manifest

    monkeypatch.setitem(pa.BUDGETS, "tiny", lambda cfg, dataset: {
        "max_steps": 4, "eval_interval": 2, "checkpoint_interval": 4})
    base = pa.load_base("tiny", _write_base(tmp_path, _tiny_cfg(vocab_size=V_WORD), dataset=WORD))
    seeds = [(1, 1), (2, 5)]
    plans = pa.build_plan(base, budget="tiny", seeds=seeds, labels=[pa.EVALUATED], counts=False)
    out = tmp_path / "out"
    path = pa.run_plan(base, plans, budget="tiny", out_root=out, device="cpu",
                       logger=logging.getLogger("paper-arms-test"))
    manifest = json.loads(path.read_text(encoding="utf-8"))
    runs = [Path(run) for run in manifest["arms"][pa.EVALUATED]["runs"]]
    assert [run.name.split("_", 1)[1] for run in runs] == ["evaluated_s1_d1", "evaluated_s2_d5"]
    for run, (seed, data_seed) in zip(runs, seeds):
        provenance = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
        assert (provenance["seed"], provenance["data_seed"]) == (seed, data_seed)
        assert pa.is_complete(run) and (run / "checkpoints").is_dir()
    assert manifest["paper_arms"]["rows"]["MLP removed"]["status"] == "not selected"
    resolved = resolve_manifest(manifest)
    assert resolved.problems == [] and resolved.arms[0].homogeneity["homogeneous"]
    assert resolved.arms[0].homogeneity["model_seed_varied"] and resolved.arms[0].homogeneity["data_seed_varied"]

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("complete (arm, seed) pairs must be skipped")

    monkeypatch.setattr(experiment, "run_vfe", refuse)
    again = pa.run_plan(base, plans, budget="tiny", out_root=out, device="cpu")
    assert json.loads(again.read_text(encoding="utf-8"))["arms"] == manifest["arms"]


def test_compute_matched_arm_calibrates_the_ladder_and_applies_the_width_rule(
        tmp_path: Path, caches: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("vfe3.baseline")
    train_baseline = pytest.importorskip("train_baseline")

    monkeypatch.setitem(pa.BUDGETS, "tiny", lambda cfg, dataset: {
        "max_steps": 4, "eval_interval": 2, "checkpoint_interval": 4})
    base = pa.load_base("tiny", _write_base(tmp_path, _tiny_cfg(vocab_size=V_WORD), dataset=WORD))
    seeds = [(1, 1)]
    labels = [pa.EVALUATED, "Matched measured compute"]
    plans = {p.label: p for p in pa.build_plan(base, budget="tiny", seeds=seeds, labels=labels, counts=False)}
    matched = plans["Matched measured compute"]
    assert matched.status == "deferred" and matched.cfg.embed_dim == 4
    out = tmp_path / "out"
    exp = pa.experiment_dir(out, "tiny", "tiny")
    logger = logging.getLogger("paper-arms-test")
    assert pa.compute_matched_config(matched, exp, calibration_steps=4, width_ladder=(1, 2),
                                     device="cpu", logger=logger) is None      # no evaluated run yet
    pa.run_plan(base, [plans[pa.EVALUATED]], budget="tiny", out_root=out, device="cpu", logger=logger)
    assert pa.ladder_widths(matched.cfg, (1, 1.5, 2)) == [4, 6, 8]
    cfg = pa.compute_matched_config(matched, exp, calibration_steps=4, width_ladder=(1, 2),
                                    device="cpu", logger=logger)
    rule = json.loads((exp / "tf_matched_compute" / "width_rule.json").read_text(encoding="utf-8"))
    assert sorted(int(w) for w in rule["calibration_runs"]) == [4, 8]
    assert [row["width"] for row in rule["candidates"]] == [4, 8]
    assert cfg.embed_dim == rule["chosen_width"] and cfg.max_steps == rule["max_steps"] > 0
    assert cfg.n_heads == cfg.embed_dim // 2                                  # head width kept
    assert (cfg.eval_interval, cfg.checkpoint_interval) == (max(1, cfg.max_steps // pa.M3_POINTS), cfg.max_steps)
    evaluated = Path(json.loads((exp / "m2_manifest.json").read_text())["evaluated_run"])
    from vfe3.baseline import measured_train_device_hours
    budget = measured_train_device_hours(evaluated)["train_hours"]
    assert rule["hours"] == pytest.approx(budget)

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("kept calibrations must be reused")

    monkeypatch.setattr(train_baseline, "calibrate_widths", refuse)
    again = pa.compute_matched_config(matched, exp, calibration_steps=4, width_ladder=(1, 2),
                                      device="cpu", logger=logger)
    assert again == cfg


def test_run_vfe_data_seed_alone_sets_the_training_batch_order(caches: Path) -> None:
    from vfe3.experiment import _loader

    cfg = _tiny_cfg()

    def order(data_seed: int) -> list:
        return [batch[0].tolist() for batch in _loader(BPE, cfg, "train", data_seed)]

    assert order(5) == order(5) and order(5) != order(6)
    evaluation = _loader(BPE, cfg, "test", 5)
    assert evaluation.dataset.pad_final and evaluation.dataset.stride == cfg.eval_stride


def test_run_vfe_reproduces_the_launcher_run_bitwise(tmp_path: Path, caches: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    r"""The same config, seed and data seed through ``train_vfe3._run_once`` and :func:`run_vfe`."""
    launcher = pytest.importorskip("train_vfe3")
    from vfe3.experiment import run_vfe

    cfg = _tiny_cfg(vocab_size=V_WORD)
    config = {key: value for key, value in serialize_config(cfg).items() if key != "seed"}
    for name, value in (("config", config), ("DATASET", WORD), ("DEVICE", "cpu"), ("MAX_TOKENS", None),
                        ("DATA_SEED", 5)):
        monkeypatch.setattr(launcher, name, value)
    logger = logging.getLogger("paper-arms-test")
    launcher._run_once(3, logger, run_root=str(tmp_path / "launcher"))
    (reference,) = [p for p in (tmp_path / "launcher").iterdir() if p.is_dir()]
    ours = run_vfe(cfg, dataset=WORD, out_root=tmp_path / "library", seed=3, data_seed=5,
                   label="tiny_s3_d5", device="cpu", logger=logger)
    assert ours.parent == tmp_path / "library" and ours.name.endswith("_tiny_s3_d5")
    a = torch.load(reference / "final_model.pt", weights_only=True)["model_state"]
    b = torch.load(ours / "final_model.pt", weights_only=True)["model_state"]
    assert a.keys() == b.keys()
    for key in a:
        assert torch.equal(a[key], b[key]), key
    summary_a = json.loads((reference / "summary.json").read_text(encoding="utf-8"))
    summary_b = json.loads((ours / "summary.json").read_text(encoding="utf-8"))
    for key in ("test_ce", "best_val_ppl", "best_step", "final_train_loss", "test_best_ce"):
        assert summary_a[key] == summary_b[key], key
    assert json.loads((ours / "provenance.json").read_text(encoding="utf-8"))["data_seed"] == 5
