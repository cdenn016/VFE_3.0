r"""Checkpoint RESUME (load side, PL8): a killed run continues from a saved checkpoint and
yields the SAME weights as an uninterrupted run.

The SAVE side (``run_artifacts.save_checkpoint``) predates this and was write-only; these
tests pin the load half: ``run_artifacts.load_checkpoint`` restores model + optimizer + RNG
+ step, and ``train(resume_from=...)`` rebuilds the per-group cosine ``LambdaLR`` at the
saved step so the continuation is numerically equivalent to a straight run.

Determinism is forced by a CONSTANT token stream: every ``TokenWindows`` window is identical,
so every batch is identical regardless of the loader's shuffle/iterator position. Both the
straight run and the resumed run therefore see the same data at every step, and the only
thing that can make their final weights differ is a missing restore leg -- model state,
optimizer momentum (exp_avg/exp_avg_sq), or the scheduler's ``last_epoch``.
"""

import copy
import json
import math
from dataclasses import asdict
from enum import IntEnum
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from vfe3.config import VFE3Config, config_from_serialized, serialize_config
from vfe3 import run_artifacts
from vfe3.data.datasets import TokenWindows
from vfe3.model.model import VFEModel
from vfe3.run_artifacts import (
    RunArtifacts,
    finalize_run,
    finalize_validation_run,
    load_checkpoint,
    semantic_config_fingerprint,
)
from vfe3.train import TrainingTerminalState, _loader_data_identity, build_optimizer, train, train_step


class _StepEnum(IntEnum):
    ONE = 1


def _const_loader(seq_len: int = 8, bs: int = 4) -> DataLoader:
    # CONSTANT stream: every window is identical, so every batch is identical regardless of
    # loader position -> the straight run and the resumed run see the same data at every step.
    base = torch.full((seq_len * 6,), 1, dtype=torch.long)
    ds = TokenWindows(base, seq_len)
    return DataLoader(ds, batch_size=bs, shuffle=False, drop_last=True)


def _cfg(**kw) -> VFE3Config:
    base = dict(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                n_e_steps=1, e_q_mu_lr=0.1, e_phi_lr=0.0, m_phi_lr=0.0,
                warmup_steps=1, max_steps=4)
    base.update(kw)
    return VFE3Config(**base)


def _params(model: torch.nn.Module):
    return [p.detach().clone() for p in model.parameters()]


def _assert_nested_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert set(actual) == set(expected)
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_nested_equal(actual_item, expected_item)
    else:
        assert actual == expected


class _SameShapeParameters(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.second = torch.nn.Parameter(torch.tensor([3.0, 4.0]))


class _TiedParameters(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.canonical = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.alias = self.canonical


class _EnabledScalerStub:
    def __init__(self, state):
        self._state = copy.deepcopy(state)

    def is_enabled(self):
        return True

    def state_dict(self):
        return copy.deepcopy(self._state)

    def load_state_dict(self, state):
        self._state = copy.deepcopy(state)


def test_config_resume_from_default_none_and_validated():
    assert VFE3Config().resume_from is None                     # off by default (pure path)
    assert VFE3Config(resume_from="ckpt.pt").resume_from == "ckpt.pt"
    with pytest.raises(ValueError):
        VFE3Config(resume_from=123)                             # not a str/path


def test_load_checkpoint_restores_model_and_returns_step(tmp_path):
    cfg = _cfg()
    torch.manual_seed(0)
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)
    art = RunArtifacts(tmp_path / "r", cfg, model)
    with torch.no_grad():                                       # make the saved state differ from a fresh init
        model.prior_bank.mu_embed.add_(0.5)
    saved = _params(model)
    art.save_checkpoint(4, model, opt, cfg)

    fresh = VFEModel(cfg)                                       # a different random init
    assert not torch.equal(fresh.prior_bank.mu_embed, model.prior_bank.mu_embed)
    step = load_checkpoint(tmp_path / "r" / "checkpoints" / "step_4.pt", fresh)
    assert step == 4
    for restored, original in zip(_params(fresh), saved):
        assert torch.equal(restored, original)                 # exact model-state restore


def test_load_checkpoint_rejects_missing_config_before_exact_resume_mutation(tmp_path):
    cfg = _cfg()
    source = VFEModel(cfg)
    with torch.no_grad():
        source.prior_bank.mu_embed.add_(0.5)
    checkpoint = RunArtifacts(tmp_path / "source", cfg, source).save_checkpoint(
        1, source, build_optimizer(source, cfg), cfg)
    bundle = torch.load(checkpoint, weights_only=True)
    bundle.pop("config")
    malformed = tmp_path / "missing-config.pt"
    torch.save(bundle, malformed)

    target = VFEModel(cfg)
    model_before = _params(target)
    cursor = {"sentinel": "unchanged"}
    torch.manual_seed(999)
    rng_before = torch.get_rng_state().clone()

    with pytest.raises(RuntimeError, match="config.*mapping.*exact resume"):
        load_checkpoint(malformed, target, restore_rng=True, data_state=cursor)

    for expected, actual in zip(model_before, _params(target)):
        assert torch.equal(expected, actual)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert cursor == {"sentinel": "unchanged"}


def test_resume_persists_saved_and_current_source_identity(tmp_path, monkeypatch):
    identity = {
        "git_sha": "a" * 40,
        "git_dirty": False,
        "git_dirty_fingerprint": None,
    }
    monkeypatch.setattr(run_artifacts, "_git_code_identity", lambda: dict(identity))
    cfg = _cfg()
    source = VFEModel(cfg)
    source_artifacts = RunArtifacts(tmp_path / "source", cfg, source)
    checkpoint = source_artifacts.save_checkpoint(
        0, source, build_optimizer(source, cfg), cfg,
    )

    restored = VFEModel(cfg)
    restored_artifacts = RunArtifacts(tmp_path / "restored", cfg, restored)
    load_checkpoint(
        checkpoint,
        restored,
        build_optimizer(restored, cfg),
        cfg=cfg,
        artifacts=restored_artifacts,
    )

    provenance = json.loads(
        (restored_artifacts.run_dir / "resume_provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["resume_mode"] == "raw_state_resume"
    assert provenance["config_restore_policy"] == "current_config_authoritative"
    assert provenance["config_drift_fields"] == []
    expected_config_fingerprint = semantic_config_fingerprint(serialize_config(cfg))
    assert provenance["saved_config_fingerprint"] == expected_config_fingerprint
    assert provenance["current_config_fingerprint"] == expected_config_fingerprint
    assert provenance["code_identity_match"] is True
    assert provenance["saved_git_identity"] == identity
    assert provenance["current_git_identity"] == identity
    assert provenance["git_identity_status"] == "stable"


@pytest.mark.parametrize("bad_step", (True, -1, 1.5, "1", _StepEnum.ONE))
def test_save_checkpoint_rejects_noninteger_or_negative_step(tmp_path, bad_step):
    cfg = _cfg()
    model = VFEModel(cfg)
    artifacts = RunArtifacts(tmp_path / "run", cfg, model)

    with pytest.raises(ValueError, match="step.*non-negative integer"):
        artifacts.save_checkpoint(bad_step, model, build_optimizer(model, cfg), cfg)

    assert not list(artifacts.ckpt_dir.glob("*.pt"))


def test_load_checkpoint_rejects_step_above_bound_before_any_mutation(tmp_path):
    cfg = _cfg()
    source = VFEModel(cfg)
    checkpoint = RunArtifacts(tmp_path / "source", cfg, source).save_checkpoint(
        2, source, build_optimizer(source, cfg), cfg)
    target = VFEModel(cfg)
    before = _params(target)
    torch.manual_seed(999)
    rng_before = torch.get_rng_state().clone()

    with pytest.raises(ValueError, match="checkpoint step 2.*max_step=1"):
        load_checkpoint(checkpoint, target, restore_rng=True, max_step=1)

    for expected, actual in zip(before, _params(target)):
        assert torch.equal(expected, actual)
    assert torch.equal(torch.get_rng_state(), rng_before)


def test_load_checkpoint_rejects_malformed_model_state_without_partial_mutation(tmp_path):
    cfg = _cfg()
    source = VFEModel(cfg)
    checkpoint = RunArtifacts(tmp_path / "source", cfg, source).save_checkpoint(
        1, source, build_optimizer(source, cfg), cfg)
    bundle = torch.load(checkpoint, weights_only=True)
    keys = list(bundle["model_state"])
    bundle["model_state"][keys[0]] = bundle["model_state"][keys[0]] + 7.0
    del bundle["model_state"][keys[-1]]
    malformed = tmp_path / "malformed-model-state.pt"
    torch.save(bundle, malformed)
    target = VFEModel(cfg)
    before = _params(target)

    with pytest.raises(RuntimeError, match="model_state keys"):
        load_checkpoint(malformed, target)

    for expected, actual in zip(before, _params(target)):
        assert torch.equal(expected, actual)


@pytest.mark.parametrize("clock_case", ("partial", "disagree", "beyond_step"))
def test_load_checkpoint_rejects_invalid_successful_update_clocks(tmp_path, clock_case):
    cfg = _cfg()
    source = VFEModel(cfg)
    checkpoint = RunArtifacts(tmp_path / "source", cfg, source).save_checkpoint(
        1, source, build_optimizer(source, cfg), cfg)
    bundle = torch.load(checkpoint, weights_only=True)
    groups = bundle["optimizer_state"]["param_groups"]
    if clock_case == "partial":
        groups[0]["successful_updates"] = 0
    elif clock_case == "disagree":
        for index, group in enumerate(groups):
            group["successful_updates"] = index % 2
    else:
        for group in groups:
            group["successful_updates"] = 2
    malformed = tmp_path / f"bad-clock-{clock_case}.pt"
    torch.save(bundle, malformed)

    target = VFEModel(cfg)
    with pytest.raises(RuntimeError, match="successful_updates"):
        load_checkpoint(malformed, target, build_optimizer(target, cfg))


def test_load_checkpoint_requires_optimizer_state_when_optimizer_is_supplied(tmp_path):
    cfg = _cfg()
    source = VFEModel(cfg)
    checkpoint = RunArtifacts(tmp_path / "source", cfg, source).save_checkpoint(
        1, source, build_optimizer(source, cfg), cfg)
    bundle = torch.load(checkpoint, weights_only=True)
    del bundle["optimizer_state"]
    malformed = tmp_path / "missing-optimizer.pt"
    torch.save(bundle, malformed)
    target = VFEModel(cfg)

    with pytest.raises(RuntimeError, match="optimizer_state"):
        load_checkpoint(malformed, target, build_optimizer(target, cfg))


def test_load_checkpoint_rejects_malformed_adam_slots_before_mutation(tmp_path):
    cfg = _cfg()
    source = VFEModel(cfg)
    optimizer = build_optimizer(source, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    tokens, targets = next(iter(_const_loader()))
    train_step(source, optimizer, scheduler, tokens, targets)
    checkpoint = RunArtifacts(tmp_path / "source", cfg, source).save_checkpoint(
        1, source, optimizer, cfg)
    bundle = torch.load(checkpoint, weights_only=True)
    populated = next(state for state in bundle["optimizer_state"]["state"].values()
                     if "exp_avg" in state)
    del populated["exp_avg"]
    malformed = tmp_path / "missing-exp-avg.pt"
    torch.save(bundle, malformed)
    target = VFEModel(cfg)
    before = _params(target)

    with pytest.raises(RuntimeError, match="optimizer_state"):
        load_checkpoint(malformed, target, build_optimizer(target, cfg))

    for expected, actual in zip(before, _params(target)):
        assert torch.equal(expected, actual)


def _populated_optimizer_checkpoint(tmp_path):
    cfg = _cfg()
    source = VFEModel(cfg)
    optimizer = build_optimizer(source, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    tokens, targets = next(iter(_const_loader()))
    train_step(source, optimizer, scheduler, tokens, targets)
    checkpoint = RunArtifacts(tmp_path / "source", cfg, source).save_checkpoint(
        1, source, optimizer, cfg)
    return cfg, source, checkpoint


def test_checkpoint_with_pre_20260927_manifest_fields_still_resumes(tmp_path):
    """Checkpoints written before 2026-09-27 carry a populated-slot manifest and a parameter-manifest
    sha256; both are ignored on load."""
    cfg, source, checkpoint = _populated_optimizer_checkpoint(tmp_path)
    bundle = torch.load(checkpoint, weights_only=True)
    assert "optimizer_populated_slot_manifest" not in bundle
    assert "sha256" not in bundle["optimizer_parameter_manifest"]
    bundle["optimizer_populated_slot_manifest"] = {"parameter_ids": [], "sha256": "0" * 64}
    bundle["optimizer_parameter_manifest"]["sha256"] = "0" * 64
    legacy = tmp_path / "pre-20260927.pt"
    torch.save(bundle, legacy)
    target = VFEModel(cfg)
    optimizer = build_optimizer(target, cfg)

    assert load_checkpoint(legacy, target, optimizer, cfg=cfg, restore_rng=False) == 1
    _assert_nested_equal(target.state_dict(), source.state_dict())


def test_optimizer_portable_preflight_accepts_cpu_snapshot_for_fused_non_cpu_parameter():
    parameter = torch.nn.Parameter(torch.empty(2, device="meta"))
    optimizer = torch.optim.AdamW([{"params": [parameter], "fused": True}])
    saved_state = optimizer.state_dict()
    parameter_id = saved_state["param_groups"][0]["params"][0]
    saved_state["state"][parameter_id] = {
        "step": torch.tensor(1.0, dtype=torch.float32),
        "exp_avg": torch.zeros(2),
        "exp_avg_sq": torch.ones(2),
    }
    validated = run_artifacts._validate_optimizer_state(saved_state, optimizer, 1)

    assert validated is saved_state


def test_load_checkpoint_rejects_same_shape_parameter_reordering_by_name_before_mutation(tmp_path):
    cfg = _cfg()
    source = _SameShapeParameters()
    source_optimizer = torch.optim.AdamW([source.first, source.second])
    checkpoint = RunArtifacts(tmp_path / "source", cfg, source).save_checkpoint(
        0, source, source_optimizer, cfg,
    )
    target = _SameShapeParameters()
    target_optimizer = torch.optim.AdamW([target.second, target.first])
    model_before = copy.deepcopy(target.state_dict())
    optimizer_before = copy.deepcopy(target_optimizer.state_dict())

    with pytest.raises(RuntimeError, match="optimizer parameter manifest.*live optimizer"):
        load_checkpoint(checkpoint, target, target_optimizer, cfg=cfg, restore_rng=False)

    _assert_nested_equal(target.state_dict(), model_before)
    _assert_nested_equal(target_optimizer.state_dict(), optimizer_before)


def test_optimizer_parameter_manifest_uses_first_registration_name_for_tied_parameter():
    model = _TiedParameters()
    optimizer = torch.optim.AdamW([model.alias])
    optimizer_state = optimizer.state_dict()
    parameter_id = optimizer_state["param_groups"][0]["params"][0]

    manifest = run_artifacts._optimizer_parameter_manifest(model, optimizer, optimizer_state)

    assert manifest["parameter_groups"] == [[{
        "parameter_id": parameter_id,
        "name": "canonical",
        "shape": [2],
    }]]


def test_realized_optimizer_failure_rolls_back_before_other_state_mutation(tmp_path, monkeypatch):
    cfg, source, checkpoint = _populated_optimizer_checkpoint(tmp_path)
    bundle = torch.load(checkpoint, weights_only=True)
    scaler_state = {
        "scale": 32.0,
        "growth_factor": 2.0,
        "backoff_factor": 0.5,
        "growth_interval": 100,
        "_growth_tracker": 3,
    }
    bundle["scaler_state"] = scaler_state
    torch.save(bundle, checkpoint)

    target = VFEModel(cfg)
    optimizer = build_optimizer(target, cfg)
    sum(parameter.sum() for parameter in target.parameters()).backward()
    optimizer.step()
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scaler = _EnabledScalerStub({**scaler_state, "scale": 8.0})
    model_before = copy.deepcopy(target.state_dict())
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    scheduler_before = copy.deepcopy(scheduler.state_dict())
    scaler_before = copy.deepcopy(scaler.state_dict())
    torch.manual_seed(9876)
    rng_before = torch.get_rng_state().clone()

    def reject_realized_state(*_args, **_kwargs):
        raise RuntimeError("injected realized optimizer failure")

    monkeypatch.setattr(
        run_artifacts, "_validate_realized_optimizer_state", reject_realized_state, raising=False,
    )
    with pytest.raises(RuntimeError, match="injected realized optimizer failure"):
        load_checkpoint(
            checkpoint,
            target,
            optimizer,
            cfg=cfg,
            scaler=scaler,
            restore_rng=True,
        )

    _assert_nested_equal(target.state_dict(), model_before)
    _assert_nested_equal(optimizer.state_dict(), optimizer_before)
    _assert_nested_equal(scheduler.state_dict(), scheduler_before)
    _assert_nested_equal(scaler.state_dict(), scaler_before)
    assert torch.equal(torch.get_rng_state(), rng_before)


def test_signed_adam_first_moment_remains_loadable(tmp_path):
    cfg, _, checkpoint = _populated_optimizer_checkpoint(tmp_path)
    bundle = torch.load(checkpoint, weights_only=True)
    populated = next(state for state in bundle["optimizer_state"]["state"].values()
                     if "exp_avg" in state)
    populated["exp_avg"] = -populated["exp_avg"].abs() - 0.25
    signed = tmp_path / "signed-first-moment.pt"
    torch.save(bundle, signed)
    target = VFEModel(cfg)
    optimizer = build_optimizer(target, cfg)

    assert load_checkpoint(signed, target, optimizer, cfg=cfg) == 1


def test_legacy_optimizer_manifest_loads_under_unchanged_execution_contract(tmp_path):
    cfg, _, checkpoint = _populated_optimizer_checkpoint(tmp_path)
    bundle = torch.load(checkpoint, weights_only=True)
    del bundle["optimizer_parameter_manifest"]
    legacy = tmp_path / "legacy-unchanged.pt"
    torch.save(bundle, legacy)
    target = VFEModel(cfg)
    optimizer = build_optimizer(target, cfg)

    assert load_checkpoint(legacy, target, optimizer, cfg=cfg) == 1
    assert optimizer.state


def test_legacy_model_only_checkpoint_does_not_require_optimizer_manifest(tmp_path):
    cfg = _cfg()
    source = VFEModel(cfg)
    checkpoint = RunArtifacts(tmp_path / "source", cfg, source).save_checkpoint(
        0, source, build_optimizer(source, cfg), cfg,
    )
    bundle = torch.load(checkpoint, weights_only=True)
    del bundle["optimizer_parameter_manifest"]
    del bundle["optimizer_state"]
    legacy = tmp_path / "legacy-model-only.pt"
    torch.save(bundle, legacy)
    target = VFEModel(cfg)

    assert load_checkpoint(legacy, target, cfg=cfg, restore_rng=False) == 0
    for saved, restored in zip(source.parameters(), target.parameters()):
        assert torch.equal(saved, restored)


@pytest.mark.parametrize("drift", ("config", "source", "missing_live_config"))
def test_legacy_optimizer_manifest_requires_unchanged_execution_contract(tmp_path, drift):
    cfg, _, checkpoint = _populated_optimizer_checkpoint(tmp_path)
    bundle = torch.load(checkpoint, weights_only=True)
    del bundle["optimizer_parameter_manifest"]
    if drift == "source":
        bundle["code_identity_sha256"] = "0" * 64
    legacy = tmp_path / f"legacy-{drift}.pt"
    torch.save(bundle, legacy)
    target = VFEModel(cfg)
    optimizer = build_optimizer(target, cfg)
    load_cfg = None if drift == "missing_live_config" else (
        _cfg(grad_clip=0.125) if drift == "config" else cfg
    )

    with pytest.raises(RuntimeError, match="legacy optimizer.*manifest"):
        load_checkpoint(legacy, target, optimizer, cfg=load_cfg)


def test_load_checkpoint_rejects_parameter_step_beyond_successful_updates(tmp_path):
    cfg, _, checkpoint = _populated_optimizer_checkpoint(tmp_path)
    bundle = torch.load(checkpoint, weights_only=True)
    populated = next(
        state for state in bundle["optimizer_state"]["state"].values()
        if "step" in state)
    populated["step"] = torch.tensor(999.0)
    malformed = tmp_path / "future-parameter-clock.pt"
    torch.save(bundle, malformed)
    target = VFEModel(cfg)
    before = _params(target)

    with pytest.raises(RuntimeError, match="AdamW step"):
        load_checkpoint(malformed, target, build_optimizer(target, cfg))

    for expected, actual in zip(before, _params(target)):
        assert torch.equal(expected, actual)


def test_load_checkpoint_rejects_invalid_rng_state_before_model_mutation(tmp_path):
    cfg = _cfg()
    source = VFEModel(cfg)
    checkpoint = RunArtifacts(tmp_path / "source", cfg, source).save_checkpoint(
        1, source, build_optimizer(source, cfg), cfg)
    bundle = torch.load(checkpoint, weights_only=True)
    bundle["rng_state"]["cpu"] = torch.zeros(1, dtype=torch.uint8)
    malformed = tmp_path / "bad-rng.pt"
    torch.save(bundle, malformed)
    target = VFEModel(cfg)
    before = _params(target)

    with pytest.raises(RuntimeError, match="rng_state"):
        load_checkpoint(malformed, target)

    for expected, actual in zip(before, _params(target)):
        assert torch.equal(expected, actual)


def test_load_checkpoint_requires_cuda_rng_states_on_an_active_cuda_host(tmp_path, monkeypatch):
    cfg = _cfg()
    source = VFEModel(cfg)
    checkpoint = RunArtifacts(tmp_path / "source", cfg, source).save_checkpoint(
        1, source, build_optimizer(source, cfg), cfg)
    bundle = torch.load(checkpoint, weights_only=True)
    bundle["rng_state"]["cuda"] = None
    malformed = tmp_path / "missing-cuda-rng.pt"
    torch.save(bundle, malformed)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    target = VFEModel(cfg)
    before = _params(target)

    with pytest.raises(RuntimeError, match="active CUDA RNG"):
        load_checkpoint(malformed, target)

    for expected, actual in zip(before, _params(target)):
        assert torch.equal(expected, actual)


def test_load_checkpoint_rejects_retired_ema_state_before_any_mutation(tmp_path):
    cfg = _cfg()
    source = VFEModel(cfg)
    checkpoint = RunArtifacts(tmp_path / "source", cfg, source).save_checkpoint(
        1, source, build_optimizer(source, cfg), cfg)
    bundle = torch.load(checkpoint, weights_only=True)
    bundle["ema_state"] = {
        "decay": 0.9,
        "shadow": {name: tensor.detach().clone()
                   for name, tensor in source.state_dict().items()},
    }
    bundle["optimizer_state"]["unexpected_topology"] = "must not be inspected first"
    malformed = tmp_path / "retired-ema.pt"
    torch.save(bundle, malformed)
    target = VFEModel(cfg)
    before = _params(target)
    optimizer = build_optimizer(target, cfg)
    optimizer_before = copy.deepcopy(optimizer.state_dict())

    with pytest.raises(RuntimeError, match="ema_state"):
        load_checkpoint(
            malformed,
            target,
            optimizer,
            cfg=cfg,
        )

    for expected, actual in zip(before, _params(target)):
        assert torch.equal(expected, actual)
    _assert_nested_equal(optimizer.state_dict(), optimizer_before)


def test_load_checkpoint_restores_optimizer_state(tmp_path):
    # Drive a real train() run so the checkpoint is written from the actual internal optimizer's
    # populated AdamW state (exp_avg/exp_avg_sq/step), then reload it into a fresh optimizer.
    cfg = _cfg(checkpoint_interval=3)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "r", cfg, model)
    loader = _const_loader()
    train(model, loader, cfg, n_steps=3, artifacts=art)
    ckpt = tmp_path / "r" / "checkpoints" / "step_3.pt"
    assert ckpt.exists()

    fresh = VFEModel(cfg)
    fresh_opt = build_optimizer(fresh, cfg)
    assert len(fresh_opt.state) == 0                            # fresh optimizer has no momentum yet
    load_checkpoint(
        ckpt,
        fresh,
        fresh_opt,
        data_state={},
        expected_data_identity=_loader_data_identity(loader, cfg.vocab_size),
    )
    assert len(fresh_opt.state) > 0                             # AdamW momentum buffers restored from the run
    # the restored 'step' counter matches the number of completed optimizer steps
    any_state = next(iter(fresh_opt.state.values()))
    assert int(any_state["step"]) == 3


def test_load_checkpoint_restamps_current_optimizer_group_metadata(tmp_path):
    saved_cfg = _cfg(m_p_mu_lr=0.03, m_p_sigma_lr=0.02, m_phi_lr=0.01,
                     weight_decay=0.2)
    saved_model = VFEModel(saved_cfg)
    saved_opt = build_optimizer(saved_model, saved_cfg)
    art = RunArtifacts(tmp_path / "saved", saved_cfg, saved_model)
    ckpt = art.save_checkpoint(1, saved_model, saved_opt, saved_cfg)

    current_cfg = _cfg(m_p_mu_lr=0.003, m_p_sigma_lr=0.002, m_phi_lr=0.001,
                       weight_decay=0.0)
    current_model = VFEModel(current_cfg)
    current_opt = build_optimizer(current_model, current_cfg)
    current_metadata = [{k: v for k, v in group.items() if k != "params"}
                        for group in current_opt.param_groups]
    current_params = [list(group["params"]) for group in current_opt.param_groups]

    load_checkpoint(ckpt, current_model, current_opt)

    for group, metadata, params in zip(current_opt.param_groups, current_metadata, current_params):
        assert {k: v for k, v in group.items() if k != "params"} == metadata
        assert len(group["params"]) == len(params)
        assert all(loaded is current for loaded, current in zip(group["params"], params))


def test_missing_checkpoint_preserves_file_not_found(tmp_path):
    model = VFEModel(_cfg())
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "missing.pt", model)


def test_resume_matches_uninterrupted_run(tmp_path):
    r"""The end-to-end equivalence: straight 4-step run == (2 steps -> checkpoint -> resume to 4).

    Pins all three restore legs at once -- if model state, optimizer momentum, OR the LR
    schedule's last_epoch is not restored, the continuation diverges and this fails."""
    cfg = _cfg(checkpoint_interval=2)

    torch.manual_seed(0)                                        # Run A: straight through
    model_a = VFEModel(cfg)
    init_a = _params(model_a)
    train(model_a, _const_loader(), cfg, n_steps=4)
    final_a = _params(model_a)
    assert any(not torch.equal(i, f) for i, f in zip(init_a, final_a))   # actually trained (non-vacuous)

    torch.manual_seed(0)                                        # Run B: train 2, checkpoint, resume to 4
    model_b = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "run", cfg, model_b)
    train(model_b, _const_loader(), cfg, n_steps=2, artifacts=art)
    ckpt = tmp_path / "run" / "checkpoints" / "step_2.pt"
    assert ckpt.exists()

    model_c = VFEModel(cfg)                                     # fresh model resumes from the checkpoint
    losses_c = train(model_c, _const_loader(), cfg, n_steps=4, resume_from=ckpt)
    final_c = _params(model_c)

    assert len(losses_c) == 2                                   # only the remaining steps 2,3 ran
    for a, c in zip(final_a, final_c):
        torch.testing.assert_close(a, c, atol=1e-6, rtol=1e-5)  # bit-equivalent continuation


def _pullback_resume_cfg(**overrides) -> VFE3Config:
    values = {
        "checkpoint_interval": 2,
        "m_phi_lr": 0.05,
        "m_phi_update_mode": "pullback_group",
        "phi_precond_mode": "pullback_per_block",
        "transport_chart_max_norm": 6.0,
        "pos_phi": "none",
    }
    values.update(overrides)
    return _cfg(**values)


def _rewrite_checkpoint_phi_config(
    checkpoint: Path,
    legacy_value: bool,
) -> dict:
    bundle = torch.load(checkpoint, weights_only=True)
    bundle["config"].pop("m_phi_update_mode")
    bundle["config"]["pos_phi_compose"] = "group_product"
    bundle["config"]["bch_pe_order"] = 4
    bundle["config"]["m_phi_natural_grad"] = legacy_value
    bundle["config"]["m_gauge_momentum"] = 0.9
    bundle["config"]["m_gauge_update_rule"] = "heavy_ball"
    torch.save(bundle, checkpoint)
    return bundle


def test_pullback_group_resume_is_step_exact_and_has_no_phi_optimizer_slots(tmp_path):
    cfg = _pullback_resume_cfg()

    torch.manual_seed(0)
    uninterrupted = VFEModel(cfg)
    phi_initial = uninterrupted.prior_bank.phi_embed.detach().clone()
    train(uninterrupted, _sequential_loader(), cfg, n_steps=4)
    final_uninterrupted = _params(uninterrupted)
    assert not torch.equal(phi_initial, uninterrupted.prior_bank.phi_embed)

    torch.manual_seed(0)
    partial = VFEModel(cfg)
    artifacts = RunArtifacts(tmp_path / "pullback", cfg, partial)
    train(partial, _sequential_loader(), cfg, n_steps=2, artifacts=artifacts)
    checkpoint = artifacts.ckpt_dir / "step_2.pt"
    bundle = torch.load(checkpoint, weights_only=True)
    optimizer_state = bundle["optimizer_state"]
    phi_ids = {
        parameter_id
        for group in optimizer_state["param_groups"]
        if group.get("pullback_group", False)
        for parameter_id in group["params"]
    }
    assert phi_ids
    assert all(not optimizer_state["state"].get(parameter_id) for parameter_id in phi_ids)
    assert all(
        not any(str(key).startswith("gauge_") for key in slot)
        for slot in optimizer_state["state"].values()
    )

    resumed = VFEModel(cfg)
    train(resumed, _sequential_loader(), cfg, n_steps=4, resume_from=checkpoint)
    for expected, actual in zip(final_uninterrupted, _params(resumed)):
        torch.testing.assert_close(expected, actual, atol=1e-6, rtol=1e-5)


def test_retired_field_checkpoint_rejects_before_any_mutation(tmp_path):
    saved_cfg = _pullback_resume_cfg(m_phi_update_mode="adamw")
    source = VFEModel(saved_cfg)
    source_optimizer = build_optimizer(source, saved_cfg)
    source_scaler = torch.amp.GradScaler(device="cpu", enabled=True, init_scale=8.0)
    checkpoint = RunArtifacts(tmp_path / "legacy-stateful", saved_cfg, source).save_checkpoint(
        0,
        source,
        source_optimizer,
        saved_cfg,
        scaler=source_scaler,
    )
    bundle = _rewrite_checkpoint_phi_config(checkpoint, True)
    bundle["optimizer_state"]["unexpected_topology"] = "must not be inspected first"
    torch.save(bundle, checkpoint)

    active_cfg = _pullback_resume_cfg()
    target = VFEModel(active_cfg)
    target_optimizer = build_optimizer(target, active_cfg)
    target_scaler = torch.amp.GradScaler(device="cpu", enabled=True, init_scale=16.0)
    cursor = {"sentinel": "unchanged"}
    model_before = _params(target)
    optimizer_state_count_before = len(target_optimizer.state)
    optimizer_parameters_before = [
        tuple(group["params"]) for group in target_optimizer.param_groups
    ]
    optimizer_clocks_before = [
        group.get("successful_updates") for group in target_optimizer.param_groups
    ]
    scaler_before = target_scaler.state_dict()
    torch.manual_seed(999)
    rng_before = torch.get_rng_state().clone()

    with pytest.raises(RuntimeError) as exc_info:
        load_checkpoint(
            checkpoint,
            target,
            target_optimizer,
            scaler=target_scaler,
            cfg=active_cfg,
            data_state=cursor,
        )

    message = str(exc_info.value)
    assert "unknown field(s)" in message
    assert "m_phi_natural_grad" in message
    for expected, actual in zip(model_before, _params(target)):
        assert torch.equal(expected, actual)
    assert len(target_optimizer.state) == optimizer_state_count_before
    assert all(
        len(group["params"]) == len(expected)
        and all(actual is original for actual, original in zip(group["params"], expected))
        for group, expected in zip(target_optimizer.param_groups, optimizer_parameters_before)
    )
    assert [
        group.get("successful_updates") for group in target_optimizer.param_groups
    ] == optimizer_clocks_before
    assert target_scaler.state_dict() == scaler_before
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert cursor == {"sentinel": "unchanged"}


def test_ambiguous_retired_policy_resume_rejects_before_any_mutation(tmp_path):
    cfg = _cfg()
    source = VFEModel(cfg)
    source_optimizer = build_optimizer(source, cfg)
    source_scaler = torch.amp.GradScaler(device="cpu", enabled=True, init_scale=8.0)
    checkpoint = RunArtifacts(tmp_path / "ambiguous-policy", cfg, source).save_checkpoint(
        0,
        source,
        source_optimizer,
        cfg,
        scaler=source_scaler,
    )
    bundle = torch.load(checkpoint, weights_only=True)
    bundle["config"]["policy_mode"] = "future_policy"
    bundle["optimizer_state"]["unexpected_topology"] = "must not be inspected first"
    torch.save(bundle, checkpoint)

    target = VFEModel(cfg)
    target_optimizer = build_optimizer(target, cfg)
    target_scaler = torch.amp.GradScaler(device="cpu", enabled=True, init_scale=16.0)
    cursor = {"sentinel": "unchanged"}
    model_before = _params(target)
    optimizer_before = copy.deepcopy(target_optimizer.state_dict())
    scaler_before = target_scaler.state_dict()
    torch.manual_seed(999)
    rng_before = torch.get_rng_state().clone()

    with pytest.raises(RuntimeError, match="unknown field"):
        load_checkpoint(
            checkpoint,
            target,
            target_optimizer,
            scaler=target_scaler,
            cfg=cfg,
            data_state=cursor,
        )

    for expected, actual in zip(model_before, _params(target)):
        assert torch.equal(expected, actual)
    _assert_nested_equal(target_optimizer.state_dict(), optimizer_before)
    assert target_scaler.state_dict() == scaler_before
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert cursor == {"sentinel": "unchanged"}


def test_retired_field_phi_config_blocks_exact_resume_without_optimizer(tmp_path):
    saved_cfg = _pullback_resume_cfg(m_phi_update_mode="adamw")
    source = VFEModel(saved_cfg)
    with torch.no_grad():
        source.prior_bank.mu_embed.add_(0.5)
    checkpoint = RunArtifacts(tmp_path / "legacy-weights", saved_cfg, source).save_checkpoint(
        0,
        source,
        build_optimizer(source, saved_cfg),
        saved_cfg,
    )
    _rewrite_checkpoint_phi_config(checkpoint, True)

    active_cfg = _pullback_resume_cfg()
    target = VFEModel(active_cfg)
    artifacts = RunArtifacts(tmp_path / "target", active_cfg, target)
    cursor = {"sentinel": "unchanged"}
    model_before = _params(target)
    files_before = frozenset(path.relative_to(artifacts.run_dir)
                             for path in artifacts.run_dir.rglob("*") if path.is_file())
    torch.manual_seed(999)
    rng_before = torch.get_rng_state().clone()

    with pytest.raises(RuntimeError, match="unknown field"):
        load_checkpoint(
            checkpoint,
            target,
            cfg=active_cfg,
            artifacts=artifacts,
            data_state=cursor,
        )

    for expected, actual in zip(model_before, _params(target)):
        assert torch.equal(expected, actual)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert cursor == {"sentinel": "unchanged"}
    assert frozenset(path.relative_to(artifacts.run_dir)
                     for path in artifacts.run_dir.rglob("*") if path.is_file()) == files_before


@pytest.mark.parametrize(
    "legacy_slot",
    [
        {"gauge_mom": "tensor"},
        {"gauge_m": "tensor", "gauge_v": "tensor", "gauge_step": 0},
    ],
)
def test_pullback_group_resume_rejects_retired_phi_optimizer_slots(tmp_path, legacy_slot):
    cfg = _pullback_resume_cfg()
    source = VFEModel(cfg)
    optimizer = build_optimizer(source, cfg)
    checkpoint = RunArtifacts(tmp_path / "retired-slots", cfg, source).save_checkpoint(
        0, source, optimizer, cfg,
    )
    bundle = torch.load(checkpoint, weights_only=True)
    group = next(
        group for group in bundle["optimizer_state"]["param_groups"]
        if group.get("pullback_group", False)
    )
    parameter_id = group["params"][0]
    parameter = next(
        parameter
        for live_group in optimizer.param_groups
        if live_group.get("pullback_group", False)
        for parameter in live_group["params"]
    )
    slot = {
        key: (torch.zeros_like(parameter) if value == "tensor" else value)
        for key, value in legacy_slot.items()
    }
    bundle["optimizer_state"]["state"][parameter_id] = slot
    torch.save(bundle, checkpoint)

    target = VFEModel(cfg)
    with pytest.raises(RuntimeError, match="unsupported parameter slots"):
        load_checkpoint(checkpoint, target, build_optimizer(target, cfg), cfg=cfg)


def test_live_pullback_group_rejects_adam_slot_when_saved_tag_is_removed(tmp_path):
    cfg = _pullback_resume_cfg()
    source = VFEModel(cfg)
    source_optimizer = build_optimizer(source, cfg)
    checkpoint = RunArtifacts(tmp_path / "untagged-pullback-group", cfg, source).save_checkpoint(
        0, source, source_optimizer, cfg,
    )
    bundle = torch.load(checkpoint, weights_only=True)
    saved_group = next(
        group for group in bundle["optimizer_state"]["param_groups"]
        if group.get("pullback_group", False)
    )
    parameter_id = saved_group["params"][0]
    parameter = next(
        parameter
        for group in source_optimizer.param_groups
        if group.get("pullback_group", False)
        for parameter in group["params"]
    )
    saved_group.pop("pullback_group")
    bundle["optimizer_state"]["state"][parameter_id] = {
        "step": torch.tensor(0.0),
        "exp_avg": torch.zeros_like(parameter),
        "exp_avg_sq": torch.zeros_like(parameter),
    }
    torch.save(bundle, checkpoint)

    target = VFEModel(cfg)
    with pytest.raises(RuntimeError, match="pullback-group parameter"):
        load_checkpoint(checkpoint, target, build_optimizer(target, cfg), cfg=cfg)


@pytest.mark.parametrize(
    "retired_key",
    ["gauge_mom", "gauge_m", "gauge_v", "gauge_step"],
)
def test_live_adamw_group_rejects_valid_slot_with_retired_extra_key(tmp_path, retired_key):
    cfg, _, checkpoint = _populated_optimizer_checkpoint(tmp_path)
    bundle = torch.load(checkpoint, weights_only=True)
    slot = next(
        state for state in bundle["optimizer_state"]["state"].values()
        if set(state) == {"step", "exp_avg", "exp_avg_sq"}
    )
    slot[retired_key] = (
        0 if retired_key == "gauge_step" else torch.zeros_like(slot["exp_avg"])
    )
    torch.save(bundle, checkpoint)

    target = VFEModel(cfg)
    with pytest.raises(RuntimeError, match="AdamW slot schema"):
        load_checkpoint(checkpoint, target, build_optimizer(target, cfg), cfg=cfg)


def _shuffled_loader(seq_len: int = 8, bs: int = 4, n: int = 480,
                      data_seed: int = 123, loader_seed: int = 0) -> DataLoader:
    # NONCONSTANT random stream + shuffle=True (RandomSampler): distinct windows, so the batch
    # sequence actually depends on the sampler's in-flight epoch permutation.
    dg = torch.Generator().manual_seed(data_seed)
    g = torch.Generator().manual_seed(loader_seed)
    base = torch.randint(0, 6, (n,), generator=dg)
    ds = TokenWindows(base.to(torch.long), seq_len)
    return DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True, generator=g)


def _sequential_loader(seq_len: int = 8, bs: int = 4, n: int = 105,
                       data_seed: int = 123, loader_seed: int | None = None) -> DataLoader:
    generator = torch.Generator().manual_seed(data_seed)
    tokens = torch.randint(0, 6, (n,), generator=generator)
    loader_generator = (
        torch.Generator().manual_seed(loader_seed) if loader_seed is not None else None)
    return DataLoader(
        TokenWindows(tokens.to(torch.long), seq_len),
        batch_size=bs,
        shuffle=False,
        drop_last=True,
        generator=loader_generator,
    )


def test_loader_data_identity_binds_full_iterator_contract() -> None:
    tokens = torch.arange(160, dtype=torch.long) % 6

    def loader(
        *,
        seq_len:   int  = 8,
        batch_size: int = 4,
        stride:    int | None = None,
        shuffle:   bool = True,
        drop_last: bool = True,
    ) -> DataLoader:
        dataset = TokenWindows(
            tokens,
            seq_len,
            stride=stride,
            pad_final=(not shuffle and not drop_last),
        )
        generator = torch.Generator().manual_seed(0) if shuffle else None
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            generator=generator,
        )

    reference = _loader_data_identity(loader(), 6)
    variants = (
        loader(batch_size=2),
        loader(seq_len=4),
        loader(stride=4),
        loader(drop_last=False),
        loader(shuffle=False),
        loader(shuffle=False, drop_last=False),
    )

    assert all(_loader_data_identity(variant, 6) != reference for variant in variants)


def test_sequential_resume_matches_uninterrupted_nonconstant_stream(tmp_path) -> None:
    cfg = _cfg(checkpoint_interval=2, max_steps=6)

    torch.manual_seed(0)
    uninterrupted = VFEModel(cfg)
    uninterrupted_losses = train(uninterrupted, _sequential_loader(), cfg, n_steps=6)
    uninterrupted_parameters = _params(uninterrupted)

    torch.manual_seed(0)
    partial = VFEModel(cfg)
    artifacts = RunArtifacts(tmp_path / "partial", cfg, partial)
    partial_losses = train(partial, _sequential_loader(), cfg, n_steps=2, artifacts=artifacts)
    checkpoint = tmp_path / "partial" / "checkpoints" / "step_2.pt"
    saved_data_state = torch.load(checkpoint, weights_only=True)["data_state"]

    assert saved_data_state is not None
    assert saved_data_state["epoch_start_generator_state"] is None
    assert saved_data_state["batches_consumed"] == 2

    resumed = VFEModel(cfg)
    resumed_losses = train(
        resumed,
        _sequential_loader(),
        cfg,
        n_steps=6,
        resume_from=checkpoint,
    )

    assert partial_losses + resumed_losses == uninterrupted_losses
    for expected, actual in zip(uninterrupted_parameters, _params(resumed)):
        assert torch.equal(expected, actual)


def test_sequential_resume_ignores_non_sampling_loader_generator(tmp_path) -> None:
    cfg = _cfg(checkpoint_interval=2, max_steps=6)

    torch.manual_seed(0)
    uninterrupted = VFEModel(cfg)
    uninterrupted_losses = train(
        uninterrupted, _sequential_loader(loader_seed=77), cfg, n_steps=6)
    uninterrupted_parameters = _params(uninterrupted)

    torch.manual_seed(0)
    partial = VFEModel(cfg)
    artifacts = RunArtifacts(tmp_path / "partial", cfg, partial)
    partial_losses = train(
        partial, _sequential_loader(loader_seed=77), cfg, n_steps=2, artifacts=artifacts)
    checkpoint = tmp_path / "partial" / "checkpoints" / "step_2.pt"

    saved_data_state = torch.load(checkpoint, weights_only=True)["data_state"]
    assert saved_data_state["epoch_start_generator_state"] is None

    resumed = VFEModel(cfg)
    resumed_losses = train(
        resumed,
        _sequential_loader(loader_seed=77),
        cfg,
        n_steps=6,
        resume_from=checkpoint,
    )

    assert partial_losses + resumed_losses == uninterrupted_losses
    for expected, actual in zip(uninterrupted_parameters, _params(resumed)):
        assert torch.equal(expected, actual)


def test_sequential_resume_replay_does_not_consume_restored_global_rng(tmp_path) -> None:
    cfg = _cfg(checkpoint_interval=2, max_steps=2)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    artifacts = RunArtifacts(tmp_path / "partial", cfg, model)
    train(model, _sequential_loader(), cfg, n_steps=2, artifacts=artifacts)
    checkpoint = tmp_path / "partial" / "checkpoints" / "step_2.pt"
    expected_rng = torch.load(checkpoint, weights_only=True)["rng_state"]["cpu"]

    torch.manual_seed(991)
    resumed = VFEModel(cfg)
    train(
        resumed,
        _sequential_loader(),
        cfg,
        n_steps=2,
        resume_from=checkpoint,
    )

    assert torch.equal(torch.get_rng_state(), expected_rng)


def test_resume_rejects_checkpoint_beyond_requested_terminal_step(tmp_path) -> None:
    cfg = _cfg(checkpoint_interval=2, max_steps=2)
    source = VFEModel(cfg)
    artifacts = RunArtifacts(tmp_path / "source", cfg, source)
    train(source, _const_loader(), cfg, n_steps=2, artifacts=artifacts)
    checkpoint = tmp_path / "source" / "checkpoints" / "step_2.pt"
    callbacks = []

    with pytest.raises(ValueError, match="checkpoint step 2.*n_steps=1"):
        train(
            VFEModel(cfg),
            _const_loader(),
            cfg,
            n_steps=1,
            resume_from=checkpoint,
            terminal_callback=lambda *args: callbacks.append(args),
        )

    assert callbacks == []


def test_resume_rejects_step_cursor_chronology_mismatch_before_model_mutation(tmp_path) -> None:
    cfg = _cfg(checkpoint_interval=2)
    loader = _sequential_loader()
    source = VFEModel(cfg)
    artifacts = RunArtifacts(tmp_path / "source", cfg, source)
    train(source, loader, cfg, n_steps=2, artifacts=artifacts)
    checkpoint = tmp_path / "source" / "checkpoints" / "step_2.pt"
    bundle = torch.load(checkpoint, weights_only=True)
    bundle["data_state"]["epoch"] = 0
    bundle["data_state"]["batches_consumed"] = 1
    malformed = tmp_path / "bad-cursor-chronology.pt"
    torch.save(bundle, malformed)
    target = VFEModel(cfg)
    before = _params(target)

    with pytest.raises(RuntimeError, match="step.*epoch.*batches_consumed"):
        load_checkpoint(
            malformed,
            target,
            data_state={},
            expected_data_identity=_loader_data_identity(loader, cfg.vocab_size),
            expected_steps_per_epoch=len(loader),
        )

    for expected, actual in zip(before, _params(target)):
        assert torch.equal(expected, actual)


def test_fresh_periodic_checkpoint_rejects_loader_without_exact_cursor_identity(tmp_path) -> None:
    cfg = _cfg(checkpoint_interval=1)
    model = VFEModel(cfg)
    tokens = torch.ones((2, 8), dtype=torch.long)
    loader = [(tokens, tokens.clone())]
    artifacts = RunArtifacts(tmp_path / "run", cfg, model)

    with pytest.raises(RuntimeError, match="exact data.*supported DataLoader"):
        train(model, loader, cfg, n_steps=1, artifacts=artifacts)

    assert not list(artifacts.ckpt_dir.glob("*.pt"))


def test_generic_iterables_allow_artifacts_until_a_checkpoint_is_due(tmp_path) -> None:
    r"""Metrics/validation artifacts do not imply a resumable data-cursor contract."""
    cfg = _cfg()  # default checkpoint interval is beyond this one-step diagnostic run
    model = VFEModel(cfg)
    tokens = torch.ones((2, 8), dtype=torch.long)
    batches = [(tokens, tokens.clone())]
    artifacts = RunArtifacts(tmp_path / "run", cfg, model)

    losses = train(
        model,
        batches,
        cfg,
        n_steps=1,
        eval_interval=1,
        val_loader=batches,
        artifacts=artifacts,
        generate_samples=False,
    )

    assert len(losses) == 1
    assert artifacts.best_path.is_file()
    assert artifacts.selection_data_identity is None
    assert not list(artifacts.ckpt_dir.glob("*.pt"))


def test_periodic_selection_binds_validation_identity_with_no_checkpoint_due(
    tmp_path,
    monkeypatch,
) -> None:
    r"""Best-model SELECTION binds its validation contract even when no data cursor is requested.

    ``checkpoint_interval`` above ``n_steps``, no resume, no terminal callback: nothing needs a
    resumable cursor, yet ``eval_interval`` still publishes ``best_model.pt`` and a finite
    ``best_val_ppl``. Binding the identity only under the cursor gate left exactly this cell with
    selected weights carrying ``selection_data_identity=None``, which ``finalize_run`` refuses to
    reload -- after the entire training budget had already been spent.
    """
    _no_git(monkeypatch)
    cfg = _cfg(checkpoint_interval=10_000, generate_embedding_figures=False)   # never due within n_steps
    model = VFEModel(cfg)
    artifacts = RunArtifacts(tmp_path / "run", cfg, model)

    train(
        model,
        _const_loader(),
        cfg,
        n_steps=1,
        eval_interval=1,
        val_loader=_eval_loader(),
        artifacts=artifacts,
        generate_samples=False,
    )

    assert artifacts.best_path.is_file()
    assert math.isfinite(artifacts.best_val_ppl)                    # selection IS live
    assert not list(artifacts.ckpt_dir.glob("*.pt"))                # ...but no cursor work was due
    expected = _loader_data_identity(_eval_loader(), cfg.vocab_size)
    assert artifacts.selection_data_identity == expected
    saved = torch.load(artifacts.best_path, weights_only=False)
    assert saved["selection_data_identity"] == expected             # the SELECTED weights carry it
    finalize_run(model, artifacts, cfg, test_loader=_const_loader())  # the reported end-of-run crash


def test_exact_random_resume_rejects_distinct_sampler_generator(tmp_path) -> None:
    cfg = _cfg(checkpoint_interval=1)
    dataset = TokenWindows(torch.arange(105, dtype=torch.long) % 6, seq_len=8)
    sampler = torch.utils.data.RandomSampler(
        dataset, generator=torch.Generator().manual_seed(1))
    loader = DataLoader(
        dataset,
        batch_size=4,
        sampler=sampler,
        drop_last=True,
        generator=torch.Generator().manual_seed(2),
    )
    model = VFEModel(cfg)

    with pytest.raises(RuntimeError, match="sampler.generator.*loader.generator"):
        train(
            model,
            loader,
            cfg,
            n_steps=1,
            artifacts=RunArtifacts(tmp_path / "run", cfg, model),
        )


def test_exact_resume_rejects_custom_collate_contract(tmp_path) -> None:
    cfg = _cfg(checkpoint_interval=1)
    dataset = TokenWindows(torch.arange(105, dtype=torch.long) % 6, seq_len=8)

    def reverse_batch(batch):
        tokens, targets = torch.utils.data.default_collate(batch)
        return tokens.flip(0), targets.flip(0)

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        drop_last=True,
        collate_fn=reverse_batch,
    )
    model = VFEModel(cfg)

    with pytest.raises(RuntimeError, match="default collate"):
        train(
            model,
            loader,
            cfg,
            n_steps=1,
            artifacts=RunArtifacts(tmp_path / "run", cfg, model),
        )


def test_shuffled_resume_matches_uninterrupted_run(tmp_path):
    r"""A shuffled six-step run is identical to three steps plus an exact resume."""
    cfg = _cfg(checkpoint_interval=3, max_steps=6)

    torch.manual_seed(0)
    model_a = VFEModel(cfg)
    losses_a = train(model_a, _shuffled_loader(n=105), cfg, n_steps=6)
    final_a = _params(model_a)

    torch.manual_seed(0)
    model_b = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "run", cfg, model_b)
    loader_b = _shuffled_loader(n=105)
    epoch_start_generator_state = loader_b.generator.get_state().clone()
    losses_b = train(model_b, loader_b, cfg, n_steps=3, artifacts=art)
    ckpt = tmp_path / "run" / "checkpoints" / "step_3.pt"
    assert ckpt.exists()
    saved_data_state = torch.load(ckpt, weights_only=True)["data_state"]
    assert set(saved_data_state) == {
        "epoch_start_generator_state", "batches_consumed", "epoch", "data_identity",
    }
    assert saved_data_state["data_identity"] == _loader_data_identity(loader_b, cfg.vocab_size)
    assert torch.equal(saved_data_state["epoch_start_generator_state"], epoch_start_generator_state)
    assert saved_data_state["batches_consumed"] == 3
    assert saved_data_state["epoch"] == 0

    model_c = VFEModel(cfg)
    resume_art = RunArtifacts(tmp_path / "resumed", cfg, model_c)
    losses_c = train(model_c, _shuffled_loader(n=105), cfg, n_steps=6,
                     artifacts=resume_art, resume_from=ckpt)

    assert len(losses_b) == len(losses_c) == 3
    assert losses_b + losses_c == losses_a
    for uninterrupted, resumed in zip(final_a, _params(model_c)):
        assert torch.equal(uninterrupted, resumed)
    resumed_data_state = torch.load(
        tmp_path / "resumed" / "checkpoints" / "step_6.pt", weights_only=True)["data_state"]
    assert resumed_data_state["epoch"] == 1
    assert resumed_data_state["batches_consumed"] == 3
    assert not torch.equal(resumed_data_state["epoch_start_generator_state"],
                           epoch_start_generator_state)


def test_exact_shuffled_resume_requires_loader_generator(tmp_path):
    cfg = _cfg(checkpoint_interval=1, max_steps=2)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    loader = _shuffled_loader()
    loader.generator = None
    loader.sampler.generator = None
    art = RunArtifacts(tmp_path / "run", cfg, model)
    with pytest.raises(RuntimeError, match="exact shuffled resume.*generator"):
        train(model, loader, cfg, n_steps=1, artifacts=art)
    assert not (tmp_path / "run" / "checkpoints" / "step_1.pt").exists()


@pytest.mark.parametrize(("field", "bad_value"), [
    ("batches_consumed", 1.5),
    ("batches_consumed", True),
    ("batches_consumed", -1),
    ("epoch", 1.5),
    ("epoch", True),
    ("epoch", -1),
])
def test_save_checkpoint_rejects_malformed_data_cursor(tmp_path, field, bad_value):
    cfg = _cfg()
    model = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "run", cfg, model)
    data_state = {
        "epoch_start_generator_state": torch.Generator().manual_seed(0).get_state(),
        "batches_consumed":            0,
        "epoch":                       0,
        "data_identity":               _loader_data_identity(_shuffled_loader(), cfg.vocab_size),
    }
    data_state[field] = bad_value

    with pytest.raises(ValueError, match=rf"{field}.*non-negative integer"):
        art.save_checkpoint(1, model, build_optimizer(model, cfg), cfg, data_state=data_state)


def test_load_checkpoint_rejects_invalid_shuffled_epoch_generator_before_mutation(tmp_path):
    cfg = _cfg()
    source = VFEModel(cfg)
    identity = _loader_data_identity(_shuffled_loader(), cfg.vocab_size)
    data_state = {
        "epoch_start_generator_state": torch.Generator().manual_seed(0).get_state(),
        "batches_consumed":            0,
        "epoch":                       0,
        "data_identity":               identity,
    }
    checkpoint = RunArtifacts(tmp_path / "source", cfg, source).save_checkpoint(
        0, source, build_optimizer(source, cfg), cfg, data_state=data_state)
    bundle = torch.load(checkpoint, weights_only=True)
    bundle["data_state"]["epoch_start_generator_state"] = torch.zeros(1, dtype=torch.uint8)
    malformed = tmp_path / "invalid-epoch-generator.pt"
    torch.save(bundle, malformed)
    target = VFEModel(cfg)
    before = _params(target)

    with pytest.raises(RuntimeError, match="epoch_start_generator_state is invalid"):
        load_checkpoint(
            malformed,
            target,
            data_state={},
            expected_data_identity=identity,
        )

    for expected, actual in zip(before, _params(target)):
        assert torch.equal(expected, actual)


def test_resume_restores_best_val_state(tmp_path):
    r"""C2 (audit 2026-07-01): best_val_ppl/best_step are bundled by save_checkpoint and restored
    into the RunArtifacts passed to load_checkpoint, so a resumed continuation with no post-resume
    improvement still reports the run-wide best."""
    cfg = _cfg()
    torch.manual_seed(0)
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)
    art = RunArtifacts(tmp_path / "a", cfg, model)
    selection_identity = _loader_data_identity(_eval_loader(), cfg.vocab_size)
    art.bind_selection_data_identity(selection_identity)
    assert art.maybe_save_best(3, model, 8.5) is True           # model-selection state to carry over
    art.save_checkpoint(4, model, opt, cfg)

    fresh = VFEModel(cfg)
    new_art = RunArtifacts(tmp_path / "b", cfg, fresh)
    new_art.bind_selection_data_identity(selection_identity)
    assert new_art.best_val_ppl == float("inf") and new_art.best_step is None
    load_checkpoint(tmp_path / "a" / "checkpoints" / "step_4.pt", fresh, artifacts=new_art)
    assert new_art.best_val_ppl == 8.5
    assert new_art.best_step == 3


def test_resume_from_cfg_field_is_picked_up(tmp_path):
    r"""cfg.resume_from (click-to-run) is honored when no explicit resume_from arg is passed."""
    cfg = _cfg(checkpoint_interval=2)
    torch.manual_seed(0)
    model_b = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "run", cfg, model_b)
    train(model_b, _const_loader(), cfg, n_steps=2, artifacts=art)
    ckpt = tmp_path / "run" / "checkpoints" / "step_2.pt"

    cfg_resume = _cfg(checkpoint_interval=2, resume_from=str(ckpt))
    model_c = VFEModel(cfg_resume)
    losses_c = train(model_c, _const_loader(), cfg_resume, n_steps=4)
    assert len(losses_c) == 2                                   # resumed from step 2 via the cfg field


def test_resume_config_drift_reports_grad_clip_change(tmp_path):
    r"""PB-15: changing only grad_clip across a resume is caught by the existing config-drift
    warning in load_checkpoint (the same mechanism test_resume_warns_on_config_drift pins for
    e_q_mu_lr), proving grad_clip participates in that comparison like any other semantic field."""
    cfg = _cfg(grad_clip=1.0)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)
    art = RunArtifacts(tmp_path / "r", cfg, model)
    path = art.save_checkpoint(2, model, opt, cfg)

    drifted_payload = serialize_config(cfg)
    drifted_payload["grad_clip"] = 0.25
    drifted = config_from_serialized(drifted_payload, source="grad-clip drift fixture")
    with pytest.warns(UserWarning, match=r"config drift.*grad_clip"):
        load_checkpoint(path, model, opt, cfg=drifted)

    # identical config (grad_clip unchanged) -> silent
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("error")
        load_checkpoint(path, model, opt, cfg=cfg)


# --------------------------------------------------------------------------- PB-03: portable best weights
#
# A cross-run-directory resume used to restore only the best_val_ppl/best_step SCALARS, so the
# selected weights (best_model.pt) never followed the checkpoint into the new run_dir; finalize then
# saw finite best metadata whose file did not exist. save_checkpoint now embeds a VALIDATED best-model
# bundle, and load_checkpoint publishes it (or a legacy sibling) into the new run, failing closed on
# any missing or semantically incompatible bundle. Selection compatibility is judged on the
# SELECTION PROJECTION of the config (architecture/objective fields), so a resume-path or output-cadence
# change cannot invalidate otherwise identical weights. Tiny CPU models throughout (embed_dim=4).


def _eval_loader(seq_len: int = 8, bs: int = 4) -> DataLoader:
    base = torch.arange(3).repeat(20)
    ds = TokenWindows(base[: seq_len * 6].long(), seq_len)
    return DataLoader(ds, batch_size=bs, shuffle=False, drop_last=True)


def _no_git(monkeypatch) -> None:
    monkeypatch.setattr(
        "vfe3.run_artifacts._git_code_identity",
        lambda *a, **k: {"git_sha": "0" * 40, "git_dirty": False, "git_dirty_fingerprint": None})


def _make_terminal_state(model, cfg) -> TrainingTerminalState:
    r"""A TrainingTerminalState from the model's current (raw) weights + a fresh optimizer + RNG."""
    opt = build_optimizer(model, cfg)
    raw = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    rng = {"cpu": torch.get_rng_state().clone(),
           "cuda": ([s.clone() for s in torch.cuda.get_rng_state_all()]
                    if torch.cuda.is_available() else None)}
    return TrainingTerminalState(
        step=int(cfg.max_steps), optimizer=opt, scaler=None,
        data_state=None, raw_model_state=raw, rng_state=rng)


def _build_embedded_best_checkpoint(run_dir, cfg, *, best_ppl=5.0, best_step=2, final_step=4):
    r"""Write a checkpoint whose embedded best bundle carries DISTINCT selected weights.

    Returns (checkpoint_path, best_state, final_state). The best weights (saved to best_model.pt at
    best_ppl) differ from the checkpoint's own model_state, so a later equality check proves the
    SELECTED weights -- not the latest weights -- were carried across the resume."""
    torch.manual_seed(0)
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)
    art = RunArtifacts(run_dir, cfg, model)
    art.bind_selection_data_identity(_loader_data_identity(_eval_loader(), cfg.vocab_size))
    with torch.no_grad():
        model.prior_bank.mu_embed.add_(0.5)                     # BEST weights
    best_state = {n: p.detach().clone() for n, p in model.named_parameters()}
    art.maybe_save_best(best_step, model, best_ppl)
    with torch.no_grad():
        model.prior_bank.mu_embed.add_(0.5)                     # FINAL weights (distinct from best)
    final_state = {n: p.detach().clone() for n, p in model.named_parameters()}
    ckpt = art.save_checkpoint(final_step, model, opt, cfg)
    return ckpt, best_state, final_state


def _strip_to_legacy(ckpt_path) -> None:
    r"""Remove the best_model_bundle field so the checkpoint reads as a pre-PB-03 (legacy) bundle."""
    bundle = torch.load(ckpt_path, weights_only=False)
    del bundle["best_model_bundle"]
    torch.save(bundle, ckpt_path)


@pytest.mark.parametrize("corruption", ("step_after_checkpoint", "empty_with_step"))
def test_resume_rejects_invalid_best_selection_metadata_before_mutation(tmp_path, corruption):
    cfg = _cfg()
    checkpoint, _, _ = _build_embedded_best_checkpoint(tmp_path / "A", cfg)
    bundle = torch.load(checkpoint, weights_only=True)
    if corruption == "step_after_checkpoint":
        bundle["best_step"] = bundle["step"] + 1
    else:
        bundle["best_val_ppl"] = float("inf")
        bundle["best_step"] = 1
        bundle["best_model_bundle"] = None
    malformed = tmp_path / f"bad-selection-{corruption}.pt"
    torch.save(bundle, malformed)
    target = VFEModel(cfg)
    before = _params(target)
    artifacts = RunArtifacts(tmp_path / "B", cfg, target)

    with pytest.raises(RuntimeError, match="best-model selection"):
        load_checkpoint(malformed, target, artifacts=artifacts)

    for expected, actual in zip(before, _params(target)):
        assert torch.equal(expected, actual)
    assert not artifacts.best_path.exists()


def test_cross_run_resume_restores_embedded_best_bundle(tmp_path):
    cfg = _cfg()
    ckpt, best_state, final_state = _build_embedded_best_checkpoint(tmp_path / "A", cfg)
    assert any(not torch.equal(best_state[n], final_state[n]) for n in best_state)   # distinct

    fresh = VFEModel(cfg)                                        # a different init, a NEW run_dir
    new_art = RunArtifacts(tmp_path / "B", cfg, fresh)
    new_art.bind_selection_data_identity(_loader_data_identity(_eval_loader(), cfg.vocab_size))
    assert new_art.best_val_ppl == float("inf") and new_art.best_step is None
    load_checkpoint(ckpt, fresh, artifacts=new_art)

    assert new_art.best_val_ppl == 5.0 and new_art.best_step == 2
    assert new_art.best_path.is_file()                          # published into the NEW run_dir
    published = torch.load(new_art.best_path, weights_only=True)["model_state"]
    for n, v in best_state.items():
        assert torch.equal(published[n], v)                     # the SELECTED (best) weights moved
    live = dict(fresh.named_parameters())
    for n, v in final_state.items():
        assert torch.equal(live[n].detach(), v)                 # the model itself got the checkpoint weights


@pytest.mark.parametrize("drift", ("code", "validation"))
def test_cross_run_resume_drops_best_selection_on_identity_drift(tmp_path, drift):
    cfg = _cfg()
    ckpt, _, final_state = _build_embedded_best_checkpoint(tmp_path / "A", cfg)
    if drift == "code":
        bundle = torch.load(ckpt, weights_only=True)
        bundle["code_identity_sha256"] = "0" * 64
        bundle["best_model_bundle"]["code_identity_sha256"] = "0" * 64
        torch.save(bundle, ckpt)

    fresh = VFEModel(cfg)
    new_art = RunArtifacts(tmp_path / "B", cfg, fresh)
    selection_loader = _eval_loader(seq_len=4) if drift == "validation" else _eval_loader()
    new_art.bind_selection_data_identity(
        _loader_data_identity(selection_loader, cfg.vocab_size))

    with pytest.warns(UserWarning, match="model selection restarts"):
        load_checkpoint(ckpt, fresh, artifacts=new_art)

    assert new_art.best_val_ppl == float("inf") and new_art.best_step is None
    assert not new_art.best_path.exists()
    live = dict(fresh.named_parameters())
    for name, value in final_state.items():
        assert torch.equal(live[name].detach(), value)          # raw training resume still succeeds


def test_legacy_cross_run_resume_imports_sibling_best_bundle(tmp_path):
    cfg = _cfg()
    ckpt, best_state, _ = _build_embedded_best_checkpoint(tmp_path / "A", cfg)
    _strip_to_legacy(ckpt)
    assert "best_model_bundle" not in torch.load(ckpt, weights_only=False)   # genuinely legacy
    assert (tmp_path / "A" / "best_model.pt").is_file()         # sibling <old_run>/best_model.pt present

    fresh = VFEModel(cfg)
    new_art = RunArtifacts(tmp_path / "B", cfg, fresh)
    new_art.bind_selection_data_identity(_loader_data_identity(_eval_loader(), cfg.vocab_size))
    load_checkpoint(ckpt, fresh, artifacts=new_art)

    assert new_art.best_val_ppl == 5.0 and new_art.best_step == 2
    assert new_art.best_path.is_file()
    published = torch.load(new_art.best_path, weights_only=True)["model_state"]
    for n, v in best_state.items():
        assert torch.equal(published[n], v)                     # sibling best imported into the new run


def test_legacy_resume_publishes_owned_preflight_snapshot_when_sibling_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg()
    checkpoint, best_state, _ = _build_embedded_best_checkpoint(tmp_path / "A", cfg)
    _strip_to_legacy(checkpoint)
    sibling = tmp_path / "A" / "best_model.pt"
    replacement = torch.load(sibling, weights_only=True)
    with torch.no_grad():
        for key, tensor in replacement["model_state"].items():
            if tensor.is_floating_point():
                replacement["model_state"][key] = tensor + 7.0

    fresh = VFEModel(cfg)
    new_art = RunArtifacts(tmp_path / "B", cfg, fresh)
    new_art.bind_selection_data_identity(_loader_data_identity(_eval_loader(), cfg.vocab_size))
    original_load_state_dict = fresh.load_state_dict

    def replace_sibling_after_preflight(state_dict, *args, **kwargs):
        torch.save(replacement, sibling)
        return original_load_state_dict(state_dict, *args, **kwargs)

    monkeypatch.setattr(fresh, "load_state_dict", replace_sibling_after_preflight)
    load_checkpoint(checkpoint, fresh, artifacts=new_art)

    replaced_sibling = torch.load(sibling, weights_only=True)["model_state"]
    for name, expected in replacement["model_state"].items():
        assert torch.equal(replaced_sibling[name], expected)
    published = torch.load(new_art.best_path, weights_only=True)["model_state"]
    for name, expected in best_state.items():
        assert torch.equal(published[name], expected)


def test_resume_without_best_weights_rejects_unreachable_selection(tmp_path):
    cfg = _cfg()
    ckpt, _, _ = _build_embedded_best_checkpoint(tmp_path / "A", cfg)
    _strip_to_legacy(ckpt)
    (tmp_path / "A" / "best_model.pt").unlink()                 # neither embedded nor sibling weights

    fresh = VFEModel(cfg)
    new_art = RunArtifacts(tmp_path / "B", cfg, fresh)
    with pytest.raises(RuntimeError, match="best-model selection"):
        load_checkpoint(ckpt, fresh, artifacts=new_art)

    assert new_art.best_val_ppl == float("inf")
    assert new_art.best_step is None
    assert not new_art.best_path.exists()


def test_checkpoint_rejects_finite_best_without_weights(tmp_path):
    cfg = _cfg()
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)
    art = RunArtifacts(tmp_path / "r", cfg, model)
    art.best_val_ppl = 5.0                                      # finite selection scalar...
    art.best_step = 2
    assert not art.best_path.exists()                          # ...with no readable best_model.pt

    with pytest.raises(RuntimeError):
        art.save_checkpoint(4, model, opt, cfg)                # integrity error -> no checkpoint


def test_finalize_rejects_best_metadata_without_best_weights(tmp_path, monkeypatch):
    _no_git(monkeypatch)
    cfg = _cfg(generate_embedding_figures=False)
    model = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "r", cfg, model)
    art.best_val_ppl = 5.0                                      # finite metadata, unreachable weights
    art.best_step = 2
    assert not art.best_path.exists()

    with pytest.raises(RuntimeError, match="no reachable weights"):
        finalize_run(model, art, cfg, test_loader=None)


def test_selection_projection_migrates_missing_defaults_and_rejects_unknown_fields():
    from vfe3.run_artifacts import _selection_semantic_config

    cfg = _cfg()
    live_projection = _selection_semantic_config(cfg)
    serialized = asdict(cfg)
    assert _selection_semantic_config(serialize_config(cfg)) == live_projection

    # The stored FULL fingerprint is a stable function of the raw mapping's key order.
    assert semantic_config_fingerprint(serialized) == semantic_config_fingerprint(
        dict(reversed(list(serialized.items()))))

    # A genuinely older mapping missing a defaulted behavior field acquires the CURRENT default.
    older = dict(serialized)
    del older["decode_tau"]
    assert "decode_tau" not in older
    assert _selection_semantic_config(older) == live_projection

    # An unknown newer field fails closed rather than being silently ignored.
    newer = dict(serialized)
    newer["a_field_from_the_future"] = 123
    assert semantic_config_fingerprint(newer) == semantic_config_fingerprint(
        dict(reversed(list(newer.items()))))
    with pytest.raises(ValueError):
        _selection_semantic_config(newer)


def test_selection_projection_excludes_checkpoint_and_figure_policy() -> None:
    from vfe3.run_artifacts import _selection_semantic_config

    assert _selection_semantic_config(_cfg(trust_resume_checkpoint=False)) == (
        _selection_semantic_config(_cfg(trust_resume_checkpoint=True))
    )
    assert _selection_semantic_config(_cfg(generate_embedding_figures=False)) == (
        _selection_semantic_config(_cfg(generate_embedding_figures=True))
    )


def test_selection_projection_rejects_a_retired_field_payload() -> None:
    from vfe3.run_artifacts import _selection_semantic_config

    legacy = asdict(_cfg())
    legacy.pop("m_phi_update_mode")
    legacy["m_phi_natural_grad"] = True
    legacy["m_gauge_momentum"] = 0.9
    legacy["m_gauge_update_rule"] = "heavy_ball"

    with pytest.raises(
        ValueError,
        match="blocks inherited model selection.*unknown field",
    ):
        _selection_semantic_config(legacy)


def test_selection_projection_grad_clip_migrates_to_default_and_differentiates():
    r"""PB-15 cross-plan regression: a raw legacy mapping predating grad_clip (simulated by
    stripping the field from asdict(VFE3Config())) migrates through _selection_semantic_config
    to the CURRENT default (grad_clip=1.0), matching the live default projection exactly -- an
    explicit non-default grad_clip=0.25 must project differently. The raw legacy mapping's own
    full fingerprint is invariant to key order (mirroring
    test_selection_projection_migrates_missing_defaults_and_rejects_unknown_fields above), and an
    unknown field still fails closed rather than being silently ignored by config_from_serialized."""
    from vfe3.run_artifacts import _selection_semantic_config

    legacy = asdict(VFE3Config())
    assert legacy["grad_clip"] == 1.0
    del legacy["grad_clip"]
    assert "grad_clip" not in legacy

    # The raw legacy mapping's own full fingerprint is a stable function of key order.
    assert semantic_config_fingerprint(legacy) == semantic_config_fingerprint(
        dict(reversed(list(legacy.items()))))

    live_default_projection = _selection_semantic_config(VFE3Config())
    assert live_default_projection["grad_clip"] == 1.0
    assert _selection_semantic_config(legacy) == live_default_projection

    explicit = dict(legacy)
    explicit["grad_clip"] = 0.25
    explicit_projection = _selection_semantic_config(explicit)
    assert explicit_projection["grad_clip"] == 0.25
    assert explicit_projection != live_default_projection

    unknown = dict(legacy)
    unknown["a_field_from_the_future"] = 123
    with pytest.raises(ValueError):
        _selection_semantic_config(unknown)


def test_resume_from_only_difference_survives_finalization(tmp_path, monkeypatch):
    _no_git(monkeypatch)
    cfg_a = _cfg(generate_embedding_figures=False)              # resume_from=None
    ckpt, best_state, final_state = _build_embedded_best_checkpoint(tmp_path / "A", cfg_a)

    # Resume into B under a config that differs ONLY in resume_from (and trust flags).
    cfg_b = _cfg(generate_embedding_figures=False, resume_from=str(ckpt))
    model_b = VFEModel(cfg_b)
    new_art = RunArtifacts(tmp_path / "B", cfg_b, model_b)
    new_art.bind_selection_data_identity(_loader_data_identity(_eval_loader(), cfg_b.vocab_size))
    load_checkpoint(ckpt, model_b, artifacts=new_art, cfg=cfg_b)
    assert new_art.best_val_ppl == 5.0                         # imported despite the resume_from diff
    assert new_art.best_path.is_file()
    res = finalize_run(model_b, new_art, cfg_b, test_loader=_const_loader())
    assert res["reloaded_best"] is False                       # test uses the final live weights
    assert res["evaluation_weights"] == "final"
    for name, value in model_b.named_parameters():
        torch.testing.assert_close(value, final_state[name], rtol=0, atol=0)

    # Control: raw resume is allowed, but a behavior-field difference cannot inherit selection.
    cfg_c = _cfg(generate_embedding_figures=False, decode_tau=2.0)
    model_c = VFEModel(cfg_c)
    art_c = RunArtifacts(tmp_path / "C", cfg_c, model_c)
    art_c.bind_selection_data_identity(_loader_data_identity(_eval_loader(), cfg_c.vocab_size))
    with pytest.warns(UserWarning, match="model selection restarts"):
        load_checkpoint(ckpt, model_c, artifacts=art_c, cfg=cfg_c)
    assert art_c.best_val_ppl == float("inf") and art_c.best_step is None


def test_stale_contract_rerun_replaces_old_unselected_best(tmp_path, monkeypatch):
    _no_git(monkeypatch)
    torch.manual_seed(0)
    cfg = _cfg(generate_embedding_figures=False)
    model = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "r", cfg, model)

    # A stale best_model.pt left by a previous cell: distinct weights, on disk, but the recomputed
    # cell begins with INFINITE in-memory best metadata (it must not silently select these weights).
    stale = VFEModel(cfg)
    with torch.no_grad():
        for p in stale.parameters():
            p.add_(3.0)
    art.maybe_save_best(1, stale, 2.0)
    stale_bytes = art.best_path.read_bytes()
    stale_state = torch.load(art.best_path, weights_only=True)["model_state"]
    art.best_val_ppl = float("inf")                            # recomputed-cell in-memory reset
    art.best_step = None

    state = _make_terminal_state(model, cfg)
    mapping = finalize_validation_run(
        model, art, cfg, _eval_loader(), losses=[1.0, 0.9],
        terminal_state=state, device=torch.device("cpu"))

    assert art.best_path.is_file()
    assert art.best_path.read_bytes() != stale_bytes           # terminal validation replaced the file
    new_state = torch.load(art.best_path, weights_only=True)["model_state"]
    assert any(not torch.equal(new_state[k], stale_state[k]) for k in stale_state)
    assert math.isfinite(mapping["primary_val_ppl"])
    assert Path(mapping["terminal_checkpoint"]).exists()       # success contract published
    assert (tmp_path / "r" / "summary.json").exists()


def test_active_block_mlp_checkpoint_roundtrip_and_topology_mismatch(tmp_path):
    """Generic strict state persistence restores active MLPs and rejects an off topology."""
    active_cfg = _cfg(use_block_mlp=True, n_layers=2)
    torch.manual_seed(31)
    source = VFEModel(active_cfg).eval()
    source_optimizer = build_optimizer(source, active_cfg)
    artifacts = RunArtifacts(tmp_path / "active", active_cfg, source)
    checkpoint = artifacts.save_checkpoint(1, source, source_optimizer, active_cfg)
    tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.no_grad():
        expected = source(tokens)

    restored = VFEModel(active_cfg).eval()
    load_checkpoint(checkpoint, restored, build_optimizer(restored, active_cfg), cfg=active_cfg)
    with torch.no_grad():
        torch.testing.assert_close(restored(tokens), expected, atol=0.0, rtol=0.0)

    inactive_cfg = _cfg(use_block_mlp=False, n_layers=2)
    inactive = VFEModel(inactive_cfg).eval()
    inactive_optimizer = build_optimizer(inactive, inactive_cfg)
    before = copy.deepcopy(inactive.state_dict())
    optimizer_before = copy.deepcopy(inactive_optimizer.state_dict())
    with pytest.raises((RuntimeError, ValueError)):
        load_checkpoint(
            checkpoint, inactive, inactive_optimizer, cfg=inactive_cfg,
        )
    _assert_nested_equal(inactive.state_dict(), before)
    _assert_nested_equal(inactive_optimizer.state_dict(), optimizer_before)
