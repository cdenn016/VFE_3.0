"""Loader migration of the selectors retired on October 2, 2026.

Each retired key is popped by ``config_from_serialized``: a saved value equal to the one the
projected routes pinned (or any value where the field was inert) loads silently, and any other value
is refused as retired. The lenient loader hands every retired key to that rule instead of dropping it.
"""
from pathlib import Path
from typing import Any

import pytest
import torch

from tests.profiles import test_config
from vfe3.config import RETIRED_CONFIG_KEYS, VFE3Config, config_from_serialized, serialize_config
from vfe3.legacy_config import MIGRATED_HISTORICAL_KEYS, lenient_config_from_serialized


def _payload(**updates: Any) -> dict[str, Any]:
    payload = serialize_config(test_config())
    payload.update(updates)
    return payload


def _load(**updates: Any) -> VFE3Config:
    return config_from_serialized(_payload(**updates), source="test")


# (key, accepted value, refused value). The accepted value is the pinned one.
PINNED = [
    ("divergence_family", "renyi", "jeffreys"),
    ("renyi_order", 1.0, 0.5),
    ("include_attention_entropy", True, False),
    ("gradient_mode", "filtering", "smoothing"),
    ("e_step_update", "gradient", "mm_exact"),
    ("e_step_gradient", "unroll", "straight_through"),
    ("oracle_unroll_grad", True, False),
    ("e_phi_lr", 0.0, 0.1),
    ("transport_mode", "flat", "regime_ii"),
]


@pytest.mark.parametrize(("key", "accepted", "refused"), PINNED)
def test_pinned_retired_key_loads_at_its_pinned_value_and_refuses_otherwise(
    key: str, accepted: object, refused: object,
) -> None:
    assert _load(**{key: accepted}) == test_config()
    with pytest.raises(ValueError, match=rf"retired {key}="):
        _load(**{key: refused})


def test_pinned_numeric_key_rejects_a_bool_stand_in() -> None:
    with pytest.raises(ValueError, match="retired renyi_order="):
        _load(renyi_order=True)
    assert _load(renyi_order=1) == test_config()


@pytest.mark.parametrize(("key", "value"), [
    ("mm_damping", 0.5),
    ("full_cov_kl_backend", "common_frame_factored"),
    ("full_cov_congruence_precision", "fp64"),
    ("full_cov_congruence_certification", True),
    ("congruence_cond_escalation", True),
    ("rope_full_gauge", True),
    ("compile_pair_kernel", True),
    ("_allow_legacy_dense_decode", True),
])
def test_inert_retired_key_loads_with_any_value(key: str, value: object) -> None:
    assert _load(**{key: value}) == test_config()


@pytest.mark.parametrize("value", [None, []])
def test_cross_couplings_accepts_none_or_empty(value: object) -> None:
    assert _load(cross_couplings=value) == test_config()


def test_cross_couplings_refuses_a_coupling() -> None:
    with pytest.raises(ValueError, match="retired cross_couplings="):
        _load(cross_couplings=[[0, 1]])


def test_mu_trust_mode_box_is_refused_only_where_the_trust_region_was_read() -> None:
    # test_config is projected_gradient + euler + e_mu_q_trust=2: the box region was live.
    with pytest.raises(ValueError, match="retired mu_trust_mode='box'"):
        _load(mu_trust_mode="box")
    assert _load(mu_trust_mode="ball") == test_config()
    assert _load(mu_trust_mode="box", e_mu_q_trust=None) == test_config(e_mu_q_trust=None)
    closed = _load(mu_trust_mode="box", relational_receiver_update="closed_form")
    assert closed == test_config(relational_receiver_update="closed_form")
    predictive = dict(attention_mode="projected_predictive", relational_receiver_law="channel_likelihood",
                      relational_receiver_update="closed_form")
    assert _load(mu_trust_mode="box", **predictive) == test_config(**predictive)


def test_e_step_mu_precond_raw_is_inert_on_predictive_and_refused_on_gradient() -> None:
    with pytest.raises(ValueError, match="retired e_step_mu_precond='raw'"):
        _load(e_step_mu_precond="raw")
    assert _load(e_step_mu_precond="fisher") == test_config()
    predictive = dict(attention_mode="projected_predictive", relational_receiver_law="channel_likelihood",
                      relational_receiver_update="closed_form")
    assert _load(e_step_mu_precond="raw", **predictive) == test_config(**predictive)


def test_m_decode_lr_migrates_into_unset_split_rates() -> None:
    cfg = _load(m_decode_lr=0.3)                                   # linear: mean-role rate only
    assert (cfg.m_decode_mu_lr, cfg.m_decode_sigma_lr) == (0.3, None)
    bank = dict(decode_mode="diagonal_chunked", decode_temperature_learned=False)
    cfg = _load(m_decode_lr=0.3, **bank)
    assert (cfg.m_decode_mu_lr, cfg.m_decode_sigma_lr) == (0.3, 0.3)
    cfg = _load(m_decode_lr=0.3, m_decode_mu_lr=0.01, **bank)
    assert (cfg.m_decode_mu_lr, cfg.m_decode_sigma_lr) == (0.01, 0.3)
    assert _load(m_decode_lr=None) == test_config()


def test_m_decode_lr_is_refused_where_it_set_a_learned_temperature_rate() -> None:
    with pytest.raises(ValueError, match="retired m_decode_lr="):
        _load(m_decode_lr=0.3, decode_mode="diagonal_chunked", decode_temperature_learned=True)
    cfg = _load(m_decode_lr=0.3, decode_mode="diagonal_chunked", decode_temperature_learned=False)
    assert cfg.m_decode_mu_lr == 0.3


@pytest.mark.parametrize(("dense", "chunked"), [
    ("diagonal", "diagonal_chunked"), ("full", "full_chunked"), ("family", "family_chunked"),
])
def test_dense_decode_modes_migrate_to_their_chunked_kernels(dense: str, chunked: str) -> None:
    family = "gaussian_full" if dense == "full" else "gaussian_diagonal"
    cfg = _load(decode_mode=dense, family=family, _allow_legacy_dense_decode=True)
    assert cfg.decode_mode == chunked
    assert "_allow_legacy_dense_decode" not in serialize_config(cfg)


def test_gaussian_diagonal_exact_migrates_to_gaussian_diagonal() -> None:
    assert _load(family="gaussian_diagonal_exact").family == "gaussian_diagonal"


def test_gaussian_frame_diagonal_is_refused_as_retired() -> None:
    with pytest.raises(ValueError, match="retired family='gaussian_frame_diagonal'"):
        _load(family="gaussian_frame_diagonal")


@pytest.mark.parametrize("mode", ["canonical_content_gauge", "canonical_content_projected"])
def test_canonical_content_encoders_are_refused_as_retired(mode: str) -> None:
    with pytest.raises(ValueError, match=f"retired encode_mode='{mode}'"):
        _load(encode_mode=mode)


@pytest.mark.parametrize("group", ["glk", "tied_block_glk", "so_k", "sp", "sp_n"])
def test_retired_gauge_groups_are_refused_on_load_and_construction(group: str) -> None:
    with pytest.raises(ValueError, match="retired"):
        _load(gauge_group=group)
    with pytest.raises(ValueError, match="retired"):
        test_config(gauge_group=group)


def test_retired_fields_are_not_constructor_arguments() -> None:
    for key in ("e_step_update", "mu_trust_mode", "renyi_order", "transport_mode", "m_decode_lr",
                "_allow_legacy_dense_decode", "decode_temperature_learned", "safe_cholesky_jitter_mode",
                "exp_fp64_norm_threshold", "transport_clamp_monitor",
                "decode_ce_stable_reduction", "decode_ranking_refinement",
                "predictive_covariance_centering", "min_lr"):
        with pytest.raises(TypeError):
            test_config(**{key: None})


def test_lenient_loader_routes_every_retired_key_through_the_strict_rule() -> None:
    assert MIGRATED_HISTORICAL_KEYS == RETIRED_CONFIG_KEYS
    assert "compile_pair_kernel" in MIGRATED_HISTORICAL_KEYS
    cfg, dropped = lenient_config_from_serialized(
        _payload(e_step_update="gradient", compile_pair_kernel=True, mm_damping=0.5, unknown_knob=1),
        source="test")
    assert cfg == test_config()
    assert dropped == {"unknown_knob": 1}
    with pytest.raises(ValueError, match="retired e_step_update='mm_exact'"):
        lenient_config_from_serialized(_payload(e_step_update="mm_exact"), source="test")


# ---- decode_temperature_learned and safe_cholesky_jitter_mode (retired later on 2026-10-02) ----

@pytest.mark.parametrize("decode_mode", ["linear", "diagonal_chunked"])
@pytest.mark.parametrize("learned", [True, False])
def test_decode_temperature_learned_loads_with_either_value(decode_mode: str, learned: bool) -> None:
    # The saved decode_log_scale is restored from the weights and held frozen, so evaluation of a
    # learned-temperature run replays its temperature exactly; resume is guarded separately below.
    assert (_load(decode_temperature_learned=learned, decode_mode=decode_mode)
            == test_config(decode_mode=decode_mode))


def test_decode_temperature_learned_rejects_a_non_boolean() -> None:
    with pytest.raises(ValueError, match="decode_temperature_learned"):
        _load(decode_temperature_learned="yes")


def test_relative_jitter_loads_on_every_family() -> None:
    assert _load(safe_cholesky_jitter_mode="relative") == test_config()
    full = _load(safe_cholesky_jitter_mode="relative", family="gaussian_full")
    assert full == test_config(family="gaussian_full")


def test_absolute_jitter_loads_only_where_it_was_inert() -> None:
    assert test_config().diagonal_covariance
    assert _load(safe_cholesky_jitter_mode="absolute") == test_config()
    with pytest.raises(ValueError, match="retired safe_cholesky_jitter_mode='absolute'"):
        _load(safe_cholesky_jitter_mode="absolute", family="gaussian_full")


def test_safe_cholesky_ridge_is_relative_to_the_diagonal_mean() -> None:
    from vfe3.numerics import safe_cholesky
    # Diagonal mean ~50 with one direction at -1e-5: an absolute ridge 1e-6 * 10^t would need
    # t = 2 (1e-4); the relative ridge 1e-6 * |diagonal mean| already repairs it at t = 0.
    M = torch.diag(torch.tensor([100.0, -1e-5], dtype=torch.float64))
    L, ok = safe_cholesky(M, rounds=1)
    assert bool(ok)
    ridge = 1e-6 * M.diagonal().mean().abs()
    expected = torch.linalg.cholesky(M + ridge * torch.eye(2, dtype=M.dtype))
    torch.testing.assert_close(L, expected, rtol=0.0, atol=0.0)


def _historical_learned_temperature_checkpoint(tmp_path: Path, decode_mode: str):
    """Write a checkpoint whose optimizer groups decode_log_scale, as the retired True runs did."""
    from vfe3.model.model import VFEModel
    from vfe3.run_artifacts import RunArtifacts
    from vfe3.train import build_optimizer, train_step

    cfg = test_config(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1, n_e_steps=1,
                      warmup_steps=1, max_steps=4, decode_mode=decode_mode)
    torch.manual_seed(0)
    source = VFEModel(cfg)
    scale = source.prior_bank.decode_log_scale
    optimizer = build_optimizer(source, cfg)
    scale.requires_grad_(True)
    sigma_group = next(group for group in optimizer.param_groups if group.get("role") == "sigma")
    sigma_group["params"].append(scale)          # the historical shared sigma/temperature group
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    tokens = torch.randint(0, 6, (2, 8), generator=torch.Generator().manual_seed(1))
    train_step(source, optimizer, scheduler, tokens, torch.roll(tokens, -1, dims=1))
    checkpoint = RunArtifacts(tmp_path / "source", cfg, source).save_checkpoint(
        1, source, optimizer, cfg)
    return cfg, source, optimizer, checkpoint


def test_resume_refuses_an_optimizer_that_trained_the_retired_temperature(tmp_path: Path) -> None:
    from vfe3.model.model import VFEModel
    from vfe3.run_artifacts import load_checkpoint
    from vfe3.train import build_optimizer

    cfg, _, optimizer, checkpoint = _historical_learned_temperature_checkpoint(
        tmp_path, "diagonal_chunked")
    assert optimizer.state[optimizer.param_groups[1]["params"][-1]]     # the temperature stepped
    target = VFEModel(cfg)
    with pytest.raises(RuntimeError, match="retired learned decode temperature"):
        load_checkpoint(checkpoint, target, build_optimizer(target, cfg), cfg=cfg, restore_rng=False)
    # Weights alone (evaluation) still load, temperature included.
    evaluated = VFEModel(cfg)
    load_checkpoint(checkpoint, evaluated, cfg=cfg, restore_rng=False)
    saved = torch.load(checkpoint, weights_only=True)["model_state"]["prior_bank.decode_log_scale"]
    assert torch.equal(evaluated.prior_bank.decode_log_scale.detach(), saved)
    assert not evaluated.prior_bank.decode_log_scale.requires_grad


def test_resume_drops_a_never_stepped_retired_temperature_exactly(tmp_path: Path) -> None:
    # Under linear decode the readout discards tau_eff, so the grouped temperature never had a slot.
    from vfe3.model.model import VFEModel
    from vfe3.run_artifacts import load_checkpoint
    from vfe3.train import build_optimizer

    cfg, source, optimizer, checkpoint = _historical_learned_temperature_checkpoint(tmp_path, "linear")
    scale = source.prior_bank.decode_log_scale
    assert not optimizer.state.get(scale)
    target = VFEModel(cfg)
    target_optimizer = build_optimizer(target, cfg)
    assert load_checkpoint(checkpoint, target, target_optimizer, cfg=cfg, restore_rng=False) == 1
    for name, parameter in source.state_dict().items():
        assert torch.equal(parameter, target.state_dict()[name])
    live = {id(p): name for name, p in target.named_parameters()}
    saved = {id(p): name for name, p in source.named_parameters()}
    restored = {live[id(p)]: state for p, state in target_optimizer.state.items()}
    original = {saved[id(p)]: state for p, state in optimizer.state.items()}
    assert set(restored) == set(original)
    for name, state in original.items():
        for slot, value in state.items():
            assert torch.equal(restored[name][slot], value)


# ---- one transport domain radius (retired later on 2026-10-02) ---------------------------------

@pytest.mark.parametrize(("key", "value"), [
    ("transport_clamp_monitor", True), ("transport_clamp_monitor", False),
    ("exp_fp64_norm_threshold", 20.0), ("exp_fp64_norm_threshold", 5.0),
])
def test_retired_exponential_policy_keys_load_with_any_value(key: str, value: object) -> None:
    assert _load(**{key: value}) == test_config()


# frame_pair_fp64_escalation was retired on 2026-10-02 and is a live opt-in field again since
# 2026-10-04: a saved value loads as saved, and a config without it loads the default False.
@pytest.mark.parametrize("value", [True, False])
def test_reinstated_frame_pair_escalation_loads_its_saved_value(value: bool) -> None:
    assert _load(frame_pair_fp64_escalation=value).frame_pair_fp64_escalation is value
    payload = _payload()
    del payload["frame_pair_fp64_escalation"]
    assert config_from_serialized(payload, source="test").frame_pair_fp64_escalation is False


def test_a_missing_chart_radius_loads_as_the_retired_clamp_radius() -> None:
    assert _load(transport_chart_max_norm=None).transport_chart_max_norm == 20.0
    assert _load(transport_chart_max_norm=18).transport_chart_max_norm == 18


@pytest.mark.parametrize("radius", [None, 0.0, -1.0, float("inf"), float("nan"), True])
def test_the_chart_radius_is_a_required_positive_finite_number(radius: object) -> None:
    with pytest.raises(ValueError, match="transport_chart_max_norm must be finite and positive"):
        test_config(transport_chart_max_norm=radius)


def test_the_mstep_interior_radius_lies_strictly_inside_the_chart() -> None:
    assert test_config(phi_mstep_max_matrix_norm=12.0, transport_chart_max_norm=18.0)
    for radius in (18.0, 25.0):
        with pytest.raises(ValueError, match="strictly below transport_chart_max_norm"):
            test_config(phi_mstep_max_matrix_norm=radius, transport_chart_max_norm=18.0)
    with pytest.raises(ValueError, match="strictly below transport_chart_max_norm"):
        _load(phi_mstep_max_matrix_norm=25.0, transport_chart_max_norm=None)


def test_a_frame_outside_the_chart_raises_instead_of_being_rescaled() -> None:
    from vfe3.model.model import VFEModel
    cfg = test_config(transport_chart_max_norm=1.0)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    with torch.no_grad():
        model.prior_bank.phi_embed.fill_(1.0)               # embedded ||M||_F = 2 * sqrt(2) > 1
    tokens = torch.tensor([[0, 1, 2]])
    with pytest.raises(ValueError, match="transport chart validity bound exceeded"):
        model(tokens, tokens)


# ---- numerical-policy hardcoding (retired later on 2026-10-02) -----------------------------------

@pytest.mark.parametrize(("key", "value"), [
    ("decode_ce_stable_reduction", True), ("decode_ce_stable_reduction", False),
    ("decode_ranking_refinement", "candidates"), ("decode_ranking_refinement", "full"),
    ("predictive_covariance_centering", True), ("predictive_covariance_centering", False),
    ("min_lr", 0), ("min_lr", 0.0),
])
def test_rounding_level_policy_keys_load_and_drop(key: str, value: object) -> None:
    assert _load(**{key: value}) == test_config()
    assert key in RETIRED_CONFIG_KEYS


@pytest.mark.parametrize(("key", "value"), [
    ("decode_ce_stable_reduction", "yes"), ("predictive_covariance_centering", 1),
    ("decode_ranking_refinement", "some"),
])
def test_rounding_level_policy_keys_refuse_an_invalid_value(key: str, value: object) -> None:
    with pytest.raises(ValueError, match=key):
        _load(**{key: value})


@pytest.mark.parametrize("value", [1e-5, 0.2, True])
def test_a_nonzero_absolute_lr_floor_is_refused(value: object) -> None:
    with pytest.raises(ValueError, match="retired min_lr="):
        _load(min_lr=value)


@pytest.mark.parametrize("value", ["always", "never"])
def test_forced_checkpoint_policies_load_as_auto(value: str) -> None:
    assert _load(decode_ce_checkpoint=value) == test_config(decode_ce_checkpoint="auto")
    assert _load(decode_ce_checkpoint="auto_aggregate").decode_ce_checkpoint == "auto_aggregate"

def test_a_saved_none_chart_radius_loads_as_the_historical_clamp_radius(monkeypatch) -> None:
    from vfe3.config import VFE3Config
    monkeypatch.setattr(VFE3Config, "transport_chart_max_norm", 50.0)
    assert _load(transport_chart_max_norm=None).transport_chart_max_norm == 20.0
