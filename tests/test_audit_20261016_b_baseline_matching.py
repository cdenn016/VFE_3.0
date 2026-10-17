r"""Audit 2026-10-16, lane B batch 4: how the conventional baselines are matched to the VFE base.

F20: the compute-matched baseline widened up to 8x on the pre-declared ladder with the base width's
hidden-matrix learning rate (``hidden_lr_scale = 1``), while the total-matched arm transfers it muP-style
(:math:`s_0 K_0 / K`, Yang et al. 2022) and declares "lr not tuned at this width". The ladder rule now
transfers the same scale (``transfer_hidden_lr=False`` reproduces the previous rule), the frozen
``width_rule.json`` accepts the field it sets, and the arm's plan records the change and the deviation.
F58: the protocol check compared the nominal ``amp_dtype``, so a base recorded with ``'bf16'`` on a route
that runs every contraction in an fp32 island (``inactive_fields`` marks it inert) was "matched" by baselines
whose blocks run under bf16 autocast, and the Linear-decoder arm made it live without saying so. The check now
compares effective precision; a difference is a declared deviation in the plan and the manifest, and a
baseline pinned to fp32 (``baseline_overrides`` ``amp_dtype=None``) passes as matched.

Every config is tiny (K = 4, two heads, two layers, vocabulary 7); nothing is trained (the evaluated run,
the calibrations and the token count are replaced by fixed numbers).
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any

import pytest

import paper_arms as pa
from tests.test_paper_arms_20260928 import _cpu_one_thread, _tiny_cfg, _write_base  # noqa: F401

LOGGER = logging.getLogger("audit-20261016-b-baseline-matching")
COMPUTE = "Matched measured compute"
LR_DEVIATION = "lr not tuned at this width"


def _bcfg(**over: Any) -> Any:
    from vfe3.baseline import BaselineConfig

    values = dict(vocab_size=7, embed_dim=4, n_heads=2, n_layers=2, mlp_expansion=2, max_seq_len=4,
                  batch_size=2, eval_stride=2, max_steps=8, warmup_steps=0, lr=3e-3, log_interval=0,
                  eval_interval=0, checkpoint_interval=0, seed=1)
    values.update(over)
    return BaselineConfig(**values)


# ---- F20: the ladder rule transfers the hidden-matrix learning rate ----------------------------

def test_the_width_ladder_rule_transfers_the_hidden_lr_to_the_chosen_width() -> None:
    from vfe3.baseline import width_ladder_rule

    taus = {4: 1e6, 8: 1e6, 16: 1e6}                     # every width affords its required tokens
    rule = width_ladder_rule(_bcfg(), hours=1.0, tokens_per_s=taus, unique_train_tokens=1000)
    cfg = rule["config"]
    assert rule["chosen_width"] == cfg.embed_dim == 16
    assert cfg.hidden_lr_scale == pytest.approx(4 / 16) and rule["hidden_lr_scale"] == cfg.hidden_lr_scale
    assert "hidden" in rule["rule"] and "K_0/K" in rule["rule"]
    scaled = width_ladder_rule(_bcfg(hidden_lr_scale=0.5), hours=1.0, tokens_per_s=taus, unique_train_tokens=1000)
    assert scaled["config"].hidden_lr_scale == pytest.approx(0.5 * 4 / 16)      # s_0 K_0 / K
    narrow = width_ladder_rule(_bcfg(), hours=1.0, tokens_per_s={4: 1e6}, unique_train_tokens=1000)
    assert narrow["config"].hidden_lr_scale == 1.0                             # the base width keeps s_0
    previous = width_ladder_rule(_bcfg(), hours=1.0, tokens_per_s=taus, unique_train_tokens=1000,
                                 transfer_hidden_lr=False)
    assert previous["config"] == dataclasses.replace(cfg, hidden_lr_scale=1.0)


def test_the_compute_matched_arm_records_the_lr_transfer_and_reloads_its_frozen_rule(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import train_baseline  # noqa: F401
    import vfe3.baseline as api
    import vfe3.data.datasets as datasets

    base = pa.load_base("tiny", _write_base(tmp_path, _tiny_cfg()))

    def plan_now() -> pa.ArmPlan:
        plan = pa.build_plan(base, budget="base_protocol", seeds=[(1, 1)], counts=False, labels=[COMPUTE])[0]
        assert plan.status == "deferred" and plan.deviations == []
        return plan

    plan = plan_now()
    assert any("hidden_lr_scale" in note for note in plan.notes)
    exp = pa.experiment_dir(tmp_path / "out", "tiny", "base_protocol")
    evaluated = tmp_path / "evaluated_run"
    with monkeypatch.context() as m:
        m.setattr(pa, "completed_run", lambda *args, **kwargs: evaluated)
        m.setattr(api, "measured_train_device_hours", lambda run: {"train_hours": 1.0, "device_name": "cpu"})
        m.setattr(pa, "_find_calibration", lambda directory, width, *args, **kwargs: directory / f"K{width}")
        m.setattr(api, "calibrated_tokens_per_s", lambda run: {"tokens_per_s": 1e6, "device_name": "cpu"})
        m.setattr(datasets, "cached_token_count", lambda dataset, split: 1000)
        cfg = pa.compute_matched_config(plan, exp, calibration_steps=4, width_ladder=(1, 2), logger=LOGGER)
        assert (cfg.embed_dim, cfg.n_heads) == (8, 4) and cfg.hidden_lr_scale == pytest.approx(0.5)
        assert plan.diff["hidden_lr_scale"] == (1.0, 0.5) and plan.diff["embed_dim"] == (4, 8)
        recorded = [d for d in plan.deviations if d.startswith(LR_DEVIATION)]
        assert len(recorded) == 1 and "4/8" in recorded[0] and "hidden_lr_scale=0.5" in recorded[0]
        # A relaunch reloads the frozen rule (it used to refuse a hidden_lr_scale it did not set itself).
        assert pa.compute_matched_config(plan, exp, calibration_steps=4, width_ladder=(1, 2), logger=LOGGER) == cfg
        assert [d for d in plan.deviations if d.startswith(LR_DEVIATION)] == recorded
        fresh = plan_now()
        assert pa.compute_matched_config(fresh, exp, calibration_steps=4, width_ladder=(1, 2), logger=LOGGER) == cfg
        assert [d for d in fresh.deviations if d.startswith(LR_DEVIATION)] == recorded
        assert fresh.diff["hidden_lr_scale"] == (1.0, 0.5)
    row = pa.build_manifest(exp, [fresh], base=base, budget="base_protocol", seeds=[(1, 1)])[
        "paper_arms"]["rows"][COMPUTE]
    assert recorded[0] in row["declared_deviations"] and row["changes"]["hidden_lr_scale"] == ["1.0", "0.5"]


# ---- F58: effective precision in the protocol check --------------------------------------------

MATCHED_PARAMS = "Matched context-processing parameters"
PRECISION = "precision:"


def test_the_protocol_check_compares_effective_precision() -> None:
    from vfe3.baseline import BaselineConfig

    inert = _tiny_cfg(amp_dtype="bf16")                       # linear decode without decode_logits_amp
    assert "amp_dtype" in inert.inactive_fields() and pa.effective_precision(inert) == "float32"
    bf16 = BaselineConfig.matched_to(inert)
    assert bf16.amp_dtype == "bf16" and pa.effective_precision(bf16) == "bf16"
    with pytest.raises(pa.ArmDefinitionError, match="amp_dtype"):
        pa.check_baseline_arm(inert, bf16)
    assert pa.check_baseline_arm(inert, bf16, declared=("amp_dtype",)) == {"amp_dtype": ("float32", "bf16")}
    assert pa.check_baseline_arm(inert, BaselineConfig.matched_to(inert, amp_dtype=None)) == {}
    live = _tiny_cfg(amp_dtype="bf16", decode_logits_amp=True)
    assert pa.effective_precision(live) == "bf16"
    assert pa.check_baseline_arm(live, BaselineConfig.matched_to(live)) == {}
    with pytest.raises(pa.ArmDefinitionError, match="amp_dtype"):
        pa.check_baseline_arm(live, BaselineConfig.matched_to(live, amp_dtype=None))
    off = _tiny_cfg()
    assert off.amp_dtype is None and pa.check_baseline_arm(off, BaselineConfig.matched_to(off)) == {}


def test_a_bf16_baseline_against_an_effective_fp32_base_declares_the_precision(tmp_path: Path) -> None:
    import train_baseline  # noqa: F401

    base = pa.load_base("bf16", _write_base(tmp_path / "bf16", _tiny_cfg(amp_dtype="bf16")))
    plan = pa.build_plan(base, budget="base_protocol", seeds=[(1, 1)], counts=False, labels=[MATCHED_PARAMS],
                         baseline_overrides={"lr": 3e-3})[0]
    assert plan.status == "ready" and plan.diff == {"amp_dtype": ("float32", "bf16")}
    declared = [d for d in plan.deviations if d.startswith(PRECISION)]
    assert len(declared) == 1 and "bf16 autocast" in declared[0] and "inert" in declared[0]
    row = pa.build_manifest(tmp_path / "exp", [plan], base=base, budget="base_protocol", seeds=[(1, 1)])[
        "paper_arms"]["rows"][MATCHED_PARAMS]
    assert row["declared_deviations"] == plan.deviations and row["changes"]["amp_dtype"] == ["'float32'", "'bf16'"]
    # Pinned to fp32, the baseline matches the base's effective precision (it was refused before).
    pinned = pa.build_plan(base, budget="base_protocol", seeds=[(1, 1)], counts=False, labels=[MATCHED_PARAMS],
                           baseline_overrides={"lr": 3e-3, "amp_dtype": None})[0]
    assert pinned.status == "ready" and pinned.cfg.amp_dtype is None
    assert pinned.diff == {} and not any(d.startswith(PRECISION) for d in pinned.deviations)


def test_the_linear_decoder_arm_declares_the_precision_it_makes_live(tmp_path: Path) -> None:
    chunked = _tiny_cfg(amp_dtype="bf16", decode_mode="diagonal_chunked", decode_logits_amp=True)
    assert pa.effective_precision(chunked) == "float32"
    base = pa.load_base("chunked", _write_base(tmp_path / "c", chunked))
    plan = pa.build_plan(base, budget="base_protocol", seeds=[(1, 1)], counts=False, labels=["Linear decoder"])[0]
    assert plan.status == "ready" and plan.diff == {"decode_mode": ("diagonal_chunked", "linear")}
    declared = [d for d in plan.deviations if d.startswith(PRECISION)]
    assert len(declared) == 1 and "decode_logits_amp" in declared[0] and "bf16" in declared[0]
    row = pa.build_manifest(tmp_path / "exp", [plan], base=base, budget="base_protocol", seeds=[(1, 1)])[
        "paper_arms"]["rows"]["Linear decoder"]
    assert row["declared_deviations"] == plan.deviations
    fp32 = pa.load_base("fp32", _write_base(tmp_path / "f", _tiny_cfg(decode_mode="diagonal_chunked")))
    plain = pa.build_plan(fp32, budget="base_protocol", seeds=[(1, 1)], counts=False, labels=["Linear decoder"])[0]
    assert plain.status == "ready" and plain.deviations == []
