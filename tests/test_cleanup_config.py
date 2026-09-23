"""Canonical config policies and structured inactive-sweep regression tests."""
from dataclasses import fields

import pytest

from vfe3.config import VFE3Config, config_from_serialized, serialize_config


def _tiny(**overrides: object) -> VFE3Config:
    return VFE3Config(**dict(vocab_size=7, embed_dim=4, n_heads=2, max_seq_len=3,
                             **overrides))


def test_retired_policy_fields_are_not_constructor_or_serialized_fields() -> None:
    names = {field.name for field in fields(VFE3Config)}
    assert not names & {"grad_clip_per_role", "exp_fp64_mode", "relational_mode", "relational_readout_mode"}
    payload = serialize_config(_tiny())
    assert payload["attention_mode"] == "transported"
    assert config_from_serialized(payload, source="round trip") == _tiny()


@pytest.mark.parametrize("field,value", [("grad_clip_per_role", True), ("exp_fp64_mode", "norm")])
def test_supported_historical_policy_is_consumed(field: str, value: object) -> None:
    payload = serialize_config(_tiny()) | {field: value}
    assert field not in serialize_config(config_from_serialized(payload, source="old run"))


@pytest.mark.parametrize("field,value", [("grad_clip_per_role", False), ("exp_fp64_mode", "dim")])
def test_retired_historical_algorithm_is_refused(field: str, value: object) -> None:
    with pytest.raises(ValueError, match="retired"):
        config_from_serialized(serialize_config(_tiny()) | {field: value}, source="old run")


@pytest.mark.parametrize("old_readout,canonical", [
    ("conditional_update", "projected_conditional"),
    ("predictive_content", "projected_predictive"),
])
def test_historical_attention_pair_migrates(old_readout: str, canonical: str) -> None:
    payload = serialize_config(_tiny())
    payload.pop("attention_mode", None)
    payload.update(relational_mode="projected_filtering", relational_readout_mode=old_readout,
                   e_step_update="gradient", e_step_gradient="unroll", oracle_unroll_grad=True,
                   skip_belief_sigma_update=True, decode_mode="linear")
    cfg = config_from_serialized(payload, source="legacy attention")
    assert cfg.attention_mode == canonical
    assert cfg.relational_mode == "projected_filtering"
    assert cfg.relational_readout_mode == old_readout
    assert "relational_mode" not in serialize_config(cfg)
    with pytest.raises(AttributeError):
        cfg.relational_mode = "off"


def test_conflicting_attention_selectors_are_refused() -> None:
    payload = serialize_config(_tiny()) | {"attention_mode": "projected_predictive",
                                          "relational_mode": "off"}
    with pytest.raises(ValueError, match="conflict"):
        config_from_serialized(payload, source="mixed generations")


def test_activity_reports_defaults_independently_of_notice_visibility() -> None:
    cfg = _tiny(use_block_mlp=False)
    assert "block_mlp_expansion" in cfg.inactive_fields()
    assert "block_mlp_expansion" not in _tiny(use_block_mlp=True).inactive_fields()
    assert "mm_damping" in cfg.inactive_fields()


def test_disabled_mlp_expansion_sweep_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    import ablation
    baseline = serialize_config(_tiny(use_block_mlp=False))
    monkeypatch.setattr(ablation, "_resolved_sweep_baseline", lambda: dict(baseline))
    monkeypatch.setitem(ablation.SWEEPS, "cleanup_mlp", {
        "param": "block_mlp_expansion", "values": [2, 4], "scale": "linear"})
    with pytest.raises(ValueError, match="block_mlp_expansion.*INERT on EVERY"):
        ablation.validate_sweeps(["cleanup_mlp"], require_construction=True)


def test_mixed_live_mlp_contrast_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    import ablation
    baseline = serialize_config(_tiny(use_block_mlp=False))
    monkeypatch.setattr(ablation, "_resolved_sweep_baseline", lambda: dict(baseline))
    monkeypatch.setitem(ablation.SWEEPS, "cleanup_mixed", {"configs": [
        {"label": "off", "use_block_mlp": False, "block_mlp_expansion": 2},
        {"label": "on", "use_block_mlp": True, "block_mlp_expansion": 4},
    ]})
    ablation.validate_sweeps(["cleanup_mixed"], require_construction=True)


@pytest.mark.parametrize("selected_inactive_contrast", [False, True])
def test_parameter_grid_checks_only_selected_contrasts(
    monkeypatch: pytest.MonkeyPatch, selected_inactive_contrast: bool,
) -> None:
    import ablation
    baseline = serialize_config(_tiny())
    monkeypatch.setattr(ablation, "_resolved_sweep_baseline", lambda: dict(baseline))
    monkeypatch.setitem(ablation.CONFIG, "target_n_params", 10)
    monkeypatch.setitem(ablation.CONFIG, "max_param_relative_deviation", 0.0)
    monkeypatch.setitem(ablation.SWEEPS, "cleanup_matched", {
        "parameter_grid": {"embed_dim": [2, 4], "mm_damping": [0.2, 0.6]},
        "match_by": "embed_dim",
    })

    def parameter_count(arm: dict[str, object]) -> int:
        wanted = 0.6 if selected_inactive_contrast and arm["embed_dim"] == 4 else 0.2
        return 10 if arm["mm_damping"] == wanted else 20

    monkeypatch.setattr(ablation, "_realized_n_params_for_overrides", parameter_count)
    if selected_inactive_contrast:
        with pytest.raises(ValueError, match="mm_damping.*INERT on EVERY selected arm"):
            ablation._parameter_match_selection("cleanup_matched")
    else:
        selected = ablation._parameter_match_selection("cleanup_matched")["selected"]
        assert len(selected) == 2
        assert {record["overrides"]["embed_dim"] for record in selected} == {2, 4}
