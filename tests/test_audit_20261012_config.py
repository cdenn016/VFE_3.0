r"""Audit 2026-10-12 configuration remediations: C6 (``kl_max=None`` resolves to 8K), C7 (one resolution point
for the None-inherits rates and decays) and item 14 (each in-process sweep cell starts from the same fp32
matmul policy).

The expected fingerprints and optimizer groups were computed by running the same calls on the code at
6b872b4, before these changes. ``tests/data/config_payload_6b872b4.json`` is ``serialize_config(test_config())``
at that commit, i.e. a saved config.json payload of that code.
"""
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

import ablation
import scaling
from tests.profiles import config_values, test_config
from vfe3 import experiment
from vfe3.config import VFE3Config, config_from_serialized, serialize_config
from vfe3.model.model import VFEModel
from vfe3.run_artifacts import semantic_config_fingerprint
from vfe3.train import build_optimizer


_PAYLOAD_6B872B4 = json.loads(
    (Path(__file__).parent / "data" / "config_payload_6b872b4.json").read_text(encoding="utf-8"))

# semantic_config_fingerprint(asdict(config_from_serialized(payload))) at 6b872b4.
_FINGERPRINTS_6B872B4 = {
    100.0: "5f0b2d2bcfa29bb66be0a846ef3a330fa6248666ff852b1ec7890a717421a54a",
    160.0: "05d570ac0a4789a8a9176bfe5aefcc2f85669c3486d3d0f4caa9ee39c8d7cfb0",
}


def _fingerprint_over_saved_fields(cfg: VFE3Config) -> str:
    r"""Fingerprint over the fields the 6b872b4 payload carries: a field added later is a schema-only change,
    which the ablation and scaling resume paths re-derive separately."""
    return semantic_config_fingerprint(
        {key: value for key, value in asdict(cfg).items() if key in _PAYLOAD_6B872B4})


# --- C6: kl_max ------------------------------------------------------------------------------------------------

def test_kl_max_none_resolves_to_eight_k_and_saves_the_number() -> None:
    cfg = test_config(kl_max=None)
    assert cfg.kl_max == 8.0 * cfg.embed_dim
    payload = serialize_config(cfg)
    assert payload["kl_max"] == 8.0 * cfg.embed_dim
    reloaded = config_from_serialized(payload, source="test")
    assert reloaded.kl_max == cfg.kl_max
    assert semantic_config_fingerprint(asdict(reloaded)) == semantic_config_fingerprint(asdict(cfg))


@pytest.mark.parametrize("kl_max", [100.0, 160.0])
def test_saved_explicit_kl_max_loads_unchanged_with_pre_change_fingerprint(kl_max: float) -> None:
    cfg = config_from_serialized(dict(_PAYLOAD_6B872B4, kl_max=kl_max), source="test")
    assert cfg.kl_max == kl_max
    assert _fingerprint_over_saved_fields(cfg) == _FINGERPRINTS_6B872B4[kl_max]


def test_saved_config_without_kl_max_keeps_the_historical_default() -> None:
    payload = dict(_PAYLOAD_6B872B4)
    payload.pop("kl_max")
    cfg = config_from_serialized(payload, source="test")
    assert cfg.kl_max == 100.0
    assert _fingerprint_over_saved_fields(cfg) == _FINGERPRINTS_6B872B4[100.0]


# --- C7: one resolution point for the inheriting rates and decays ------------------------------------------------

def test_resolved_hyperparameter_follows_the_inheritance_chain() -> None:
    cfg = test_config(decode_mu_weight_decay=None, mu_weight_decay=None, weight_decay=0.05,
                      m_layernorm_lr=None, m_pos_phi_lr=None)
    assert cfg.resolved_hyperparameter("decode_mu_weight_decay") == 0.05
    assert cfg.inherits_hyperparameter("decode_mu_weight_decay", "weight_decay")
    assert cfg.resolved_hyperparameter("m_layernorm_lr") == cfg.m_p_mu_lr
    assert cfg.resolved_hyperparameter("m_pos_phi_lr") == cfg.m_phi_lr
    assert cfg.m_layernorm_lr is None and cfg.decode_mu_weight_decay is None   # the fields keep None
    explicit = test_config(mu_weight_decay=0.0, decode_mu_weight_decay=None)
    assert explicit.resolved_hyperparameter("decode_mu_weight_decay") == 0.0
    assert not explicit.inherits_hyperparameter("decode_mu_weight_decay", "weight_decay")


# (role, lr_aux_role, numel, lr, weight_decay) per optimizer group, from build_optimizer at 6b872b4.
_GROUPS_6B872B4 = {
    "all_none": ({}, [
        ("mu", None, 28, 0.025, 0.05), ("sigma", None, 28, 0.0025, 0.05), ("phi", None, 56, 0.015, 0.065),
        ("mu", None, 28, 0.025, 0.05), ("mu", None, 7, 0.025, 0.0), ("mu", "block_mlp", 292, 0.025, 0.05),
        ("mu", "relational_score", 32, 0.025, 0.0), ("mu", "relational_content", 32, 0.025, 0.0),
        ("phi", None, 24, 0.015, 0.065), ("mu", "layernorm", 8, 0.025, 0.0)]),
    "explicit": (dict(m_decode_mu_lr=0.011, m_block_mlp_lr=0.012, m_layernorm_lr=0.013, m_pos_phi_lr=0.014,
                      m_relational_score_lr=0.016, m_relational_content_lr=0.017, mu_weight_decay=0.0,
                      sigma_weight_decay=0.02, decode_mu_weight_decay=0.03), [
        ("mu", None, 28, 0.025, 0.0), ("sigma", None, 28, 0.0025, 0.02), ("phi", None, 56, 0.015, 0.065),
        ("mu", None, 28, 0.011, 0.03), ("mu", None, 7, 0.011, 0.0), ("mu", "block_mlp", 292, 0.012, 0.05),
        ("mu", "relational_score", 32, 0.016, 0.0), ("mu", "relational_content", 32, 0.017, 0.0),
        ("phi", None, 24, 0.014, 0.065), ("mu", "layernorm", 8, 0.013, 0.0)]),
    "untied_frozen_hidden_bias": (dict(decode_mode="diagonal_chunked", untie_decode_bank=True,
                                       per_type_hidden_bias=True, gauge_transport="frozen", mu_weight_decay=0.04,
                                       m_decode_sigma_lr=None, pos_phi="none"), [
        ("mu", None, 28, 0.025, 0.04), ("sigma", None, 28, 0.0025, 0.05), ("phi", None, 56, 0.0, 0.065),
        ("mu", None, 28, 0.025, 0.04), ("sigma", None, 28, 0.0025, 0.05), ("mu", "block_mlp", 292, 0.025, 0.05),
        ("mu", "relational_score", 32, 0.025, 0.0), ("mu", "relational_content", 32, 0.025, 0.0),
        ("phi", None, 56, 0.015, 0.065), ("mu", "layernorm", 8, 0.025, 0.0)]),
    "untied_hidden_bias_explicit": (dict(decode_mode="diagonal_chunked", untie_decode_bank=True,
                                         per_type_hidden_bias=True, m_per_type_hidden_bias_lr=0.021,
                                         m_decode_sigma_lr=0.022, m_decode_mu_lr=0.023, sigma_weight_decay=0.0,
                                         decode_mu_weight_decay=0.0), [
        ("mu", None, 28, 0.025, 0.05), ("sigma", None, 28, 0.0025, 0.0), ("phi", None, 56, 0.015, 0.065),
        ("mu", None, 28, 0.023, 0.0), ("sigma", None, 28, 0.022, 0.0), ("mu", "block_mlp", 292, 0.025, 0.05),
        ("mu", "relational_score", 32, 0.025, 0.0), ("mu", "relational_content", 32, 0.025, 0.0),
        ("phi", None, 24, 0.015, 0.065), ("phi", None, 56, 0.021, 0.065), ("mu", "layernorm", 8, 0.025, 0.0)]),
}

# Fingerprint over the 6b872b4 fields of each case's config at 6b872b4 (the None fields stay None).
_CASE_FINGERPRINTS_6B872B4 = {
    "all_none": "5f0b2d2bcfa29bb66be0a846ef3a330fa6248666ff852b1ec7890a717421a54a",
    "explicit": "1b222f3ccf5da68c81f3ccc2e12be30701dfd03f91fde966aff5e8b4cc6b0a85",
    "untied_frozen_hidden_bias": "a479cfa0d668f04327fc972ba43d1683f5fd0dda4fa9c4d45c2bedcc3ddddbf9",
    "untied_hidden_bias_explicit": "0e877550658f6f0f9313a7818d02898a6189394615fc958211b367d9f0ef17c6",
}


@pytest.mark.parametrize("case", sorted(_GROUPS_6B872B4))
def test_build_optimizer_rates_and_decays_match_pre_change_code(case: str) -> None:
    overrides, expected = _GROUPS_6B872B4[case]
    cfg = test_config(**overrides)
    optimizer = build_optimizer(VFEModel(cfg), cfg)
    realized = [(group.get("role"), group.get("lr_aux_role"), sum(p.numel() for p in group["params"]),
                 group["lr"], group["weight_decay"]) for group in optimizer.param_groups]
    assert realized == expected
    assert _fingerprint_over_saved_fields(cfg) == _CASE_FINGERPRINTS_6B872B4[case]


# --- #14: each in-process cell starts from the same fp32 matmul policy -------------------------------------------

class _StopCell(Exception):
    """Raised by the patched model constructor once the cell has pinned (or left) its matmul policy."""


def _observing_model(observed: list):
    def build(cfg: VFE3Config) -> VFEModel:
        VFEModel(cfg)                                        # the real constructor applies the cell's policy
        observed.append(torch.get_float32_matmul_precision())
        raise _StopCell
    return build


def _run_ablation_cell(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, label: str, overrides: dict) -> None:
    ablation.run_single(label, overrides, tmp_path / label, dataset="tiny", device=torch.device("cpu"), seed=1)


def _run_scaling_cell(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, label: str, overrides: dict) -> None:
    run_dir = tmp_path / label
    run_dir.mkdir()
    scaling.run_cell({"label": label, "route": "width", "scale_knob": "embed_dim", "overrides": overrides},
                     run_dir, 1, dataset="tiny", device=torch.device("cpu"))


def _run_experiment_arm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, label: str, overrides: dict) -> None:
    experiment.run_vfe(test_config(**overrides), dataset="tiny", out_root=tmp_path, seed=1, data_seed=1,
                       label=label, device="cpu")


@pytest.mark.parametrize("runner", [_run_ablation_cell, _run_scaling_cell, _run_experiment_arm],
                         ids=["ablation", "scaling", "paper_arms"])
def test_none_precision_cell_does_not_inherit_the_previous_cell_policy(tmp_path, monkeypatch, runner) -> None:
    baseline = config_values(fp32_matmul_precision=None)
    monkeypatch.setattr(ablation, "BASELINE_CONFIG", baseline)
    monkeypatch.setattr(scaling, "BASELINE", baseline)
    monkeypatch.setattr(scaling, "CONFIG", {**scaling.CONFIG, "resume": False})
    observed: list = []
    monkeypatch.setattr(ablation, "VFEModel", _observing_model(observed))
    monkeypatch.setattr(scaling, "VFEModel", _observing_model(observed))
    monkeypatch.setattr("vfe3.model.model.VFEModel", _observing_model(observed))
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        with pytest.raises(_StopCell):
            runner(tmp_path, monkeypatch, "first", {"fp32_matmul_precision": "medium"})
        assert torch.get_float32_matmul_precision() == "highest"
        with pytest.raises(_StopCell):
            runner(tmp_path, monkeypatch, "second", {"fp32_matmul_precision": None})
    finally:
        torch.set_float32_matmul_precision(previous)
    assert observed == ["medium", "highest"]
