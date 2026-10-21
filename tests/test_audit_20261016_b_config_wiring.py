r"""Audit 2026-10-16, lane B batch 1: the paper-arm base loader, two paper arms, and the joint search config.

F1 and F35: :func:`paper_arms.load_base` lists every field the loader set for an absent key (a historical pin,
a conditional migration, or a selector merged from recorded historical keys) under ``pinned`` or
``migrated``, never under ``defaulted``, while a field ``VFE3Config.__post_init__`` resolves or coerces from
the record's own fields stays a default.
F5: on a base whose block LayerNorm pushes the covariance forward on the diagonal family, the two arms that
take the block out of the pushforward's domain (the intertwiner arm replaces the LayerNorm; the
reference-coordinate arm has no diagonal reference pushforward) declare the passthrough they need instead of
resolving invalid; ``parameter_optimization.config`` spells the only value its norm admits.

Every config is tiny (K = 4, two heads, two layers, vocabulary 7); nothing is trained.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

import paper_arms as pa
import vfe3.legacy_config as legacy_config
from tests.profiles import test_config
from vfe3.config import VFE3Config, serialize_config

V = 7
REFERENCE_ARM = "LayerNorm and MLP in reference coordinates"
INTERTWINER_ARM = "Intertwiner-constrained nonlinear block"


def _tiny_cfg(**over: Any) -> VFE3Config:
    values: Dict[str, Any] = dict(
        vocab_size=V, embed_dim=4, n_heads=2, batch_size=2, max_seq_len=4, eval_stride=2, n_layers=2,
        max_steps=4, warmup_steps=0, log_interval=0, eval_interval=2, checkpoint_interval=4,
        family="gaussian_diagonal", gauge_group="block_glk", attention_mode="projected_predictive",
        attention_residual=True, decode_mode="linear", use_block_mlp=True, block_mlp_expansion=2,
        generate_embedding_figures=False, sigma_weight_decay=0.2, pos_phi="learned", seed=0,
        norm_type_block="layernorm", layernorm_affine=True, skip_belief_sigma_update=True)
    values.update(over)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return test_config(**values)


def _write_base(
    root:  Path,
    cfg:   VFE3Config,

    *,
    extra: Optional[Dict[str, Any]] = None,
    drop:  tuple = (),
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    config = {key: value for key, value in serialize_config(cfg).items() if key not in drop}
    config.update(extra or {})
    (root / "config.json").write_text(json.dumps({"config": config, "dataset": "wikitext-103"}))
    return root


def _arm(base: pa.Base, label: str) -> pa.ArmPlan:
    return pa.resolve_arm(base, label, pa.ARMS[label], budget="base_protocol", seeds=[(1, 1)], counts=False)


# ---- F35: selectors merged from recorded historical keys --------------------------------------------

def test_load_base_lists_selectors_merged_from_recorded_keys_as_migrated(tmp_path: Path) -> None:
    legacy = {"pos_rotation": "none", "rope_space": "frame",
              "projected_cov_condition_recovery": False, "projected_factor_route": "gram",
              "frame_pair_fp64_escalation": False, "frame_pair_condition_tol": 1e-4}
    merged = ("rope_mode", "projected_factorization", "frame_pair_escalation_tol")
    base = pa.load_base("tiny", _write_base(tmp_path, _tiny_cfg(), extra=legacy, drop=merged))
    assert base.migrated == {name: getattr(base.cfg, name) for name in merged}
    assert not set(merged) & set(base.defaulted)
    assert not set(merged) & set(base.pinned)
    # A record that spells the surviving selectors has nothing migrated.
    assert pa.load_base("tiny", _write_base(tmp_path / "current", _tiny_cfg())).migrated == {}


# ---- F1: a conditional migration of an absent field is reported, a coercion is not ------------------

def _assert_absent_field_listed(base: Any, name: str) -> None:
    r"""``name``, absent from the record, is listed as defaulted when its loaded value is the current
    ``VFE3Config`` default and as pinned (with that value) otherwise."""
    value = getattr(base.cfg, name)
    if value == VFE3Config.__dataclass_fields__[name].default:
        assert name in base.defaulted and name not in base.pinned, name
    else:
        assert base.pinned.get(name, object()) == value and name not in base.defaulted, name


def test_load_base_reports_a_conditional_migration_of_an_absent_field_as_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vfe3.config import config_from_serialized

    def loader(payload: Any, *, source: str) -> VFE3Config:
        # A loader migration keyed on another absent field, the form of the pre-2026-10-12 floor migration.
        payload = dict(payload)
        if "predictive_covariance_floor_mode" not in payload and "phi_frame_condition_max" not in payload:
            payload["predictive_covariance_floor_mode"] = "relative"
        return config_from_serialized(payload, source=source)

    monkeypatch.setattr(legacy_config, "config_from_serialized", loader)
    cfg = _tiny_cfg(predictive_covariance="moment_matched")
    old = pa.load_base("old", _write_base(tmp_path / "old", cfg,
                                          drop=("predictive_covariance_floor_mode", "phi_frame_condition_max")))
    assert old.cfg.predictive_covariance_floor_mode == "relative"
    assert old.pinned.get("predictive_covariance_floor_mode") == "relative"
    assert "predictive_covariance_floor_mode" not in old.defaulted
    # An absent field the loader fills with its introduction value is listed as defaulted while that value is the
    # current default, and as pinned once the default has changed (whatever vfe3/config.py holds now).
    _assert_absent_field_listed(old, "phi_frame_condition_max")
    # The newer era (the marker field recorded) loads the floor's introduction value, "prior".
    new = pa.load_base("new", _write_base(tmp_path / "new", cfg, drop=("predictive_covariance_floor_mode",)))
    assert new.cfg.predictive_covariance_floor_mode == "prior"
    _assert_absent_field_listed(new, "predictive_covariance_floor_mode")
    assert set(new.pinned) <= {"predictive_covariance_floor_mode"}


def test_load_base_keeps_a_post_init_coercion_of_an_absent_field_as_defaulted(tmp_path: Path) -> None:
    # gauge_transport='off' forces phi_scale=0 from the record's own fields; the loader did not choose it.
    base = pa.load_base("tiny", _write_base(tmp_path, _tiny_cfg(gauge_transport="off"), drop=("phi_scale",)))
    assert base.cfg.phi_scale == 0.0 and VFE3Config.phi_scale != 0.0
    assert "phi_scale" in base.defaulted and base.pinned == {}


# ---- F5: the two arms that leave the LayerNorm covariance pushforward's domain ----------------------

def test_layernorm_replacing_and_reference_arms_build_on_a_diagonal_pushforward_base(tmp_path: Path) -> None:
    base = pa.load_base("tiny", _write_base(tmp_path, _tiny_cfg(block_norm_covariance="pushforward")))
    assert (base.cfg.block_norm_covariance, base.cfg.family) == ("pushforward", "gaussian_diagonal")
    inert = "declared block_norm_covariance='passthrough' is inert here"
    for label in (REFERENCE_ARM, INTERTWINER_ARM):
        plan = _arm(base, label)
        assert plan.status == "ready", (label, plan.reason)
        assert plan.diff["block_norm_covariance"] == ("pushforward", "passthrough"), label
    # Under the intertwiner's Mahalanobis norm the field is inert; in reference coordinates on the diagonal family
    # the LayerNorm reads it (only 'passthrough' is admitted there), so that arm's passthrough is a real declared
    # change, not an inert one (the lane A1 and lane B fixes of F5 together).
    assert any(note.startswith(inert) for note in _arm(base, INTERTWINER_ARM).notes)
    assert not any(note.startswith(inert) for note in _arm(base, REFERENCE_ARM).notes)
    assert _arm(base, INTERTWINER_ARM).cfg.norm_type_block == "mahalanobis"
    assert _arm(base, REFERENCE_ARM).cfg.block_coordinates == "reference"


def test_the_two_arms_keep_the_base_block_norm_covariance_where_it_stays_read(tmp_path: Path) -> None:
    passthrough = pa.load_base("p", _write_base(tmp_path / "p", _tiny_cfg()))
    for label in (REFERENCE_ARM, INTERTWINER_ARM):
        plan = _arm(passthrough, label)
        assert plan.status == "ready" and "block_norm_covariance" not in plan.diff, label
    # The reference pushforward exists on the head-block family, so the reference arm keeps it there.
    block = pa.load_base("b", _write_base(tmp_path / "b", _tiny_cfg(family="gaussian_block",
                                                                     block_norm_covariance="pushforward")))
    plan = _arm(block, REFERENCE_ARM)
    assert plan.status == "ready", plan.reason
    assert plan.cfg.block_norm_covariance == "pushforward" and set(plan.diff) == {"block_coordinates"}


def test_parameter_optimization_config_does_not_inherit_the_block_norm_covariance_default() -> None:
    import parameter_optimization as po

    # The driver's own keys are written last, so this builds the driver config under each block_norm_covariance
    # default. The driver's values are the user's (2026-10-21): a driver that spells the field keeps its value under
    # every default; one that leaves it to the default must be valid under the default it runs with.
    spelled = "block_norm_covariance" in po.config
    current = VFE3Config.__dataclass_fields__["block_norm_covariance"].default
    for default in (("passthrough", "pushforward", "pushforward_moment") if spelled else (current,)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = VFE3Config(**{"block_norm_covariance": default, **po.config})
        assert cfg.block_norm_covariance == po.config.get("block_norm_covariance", current), default
