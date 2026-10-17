r"""The spectral (Daleckii-Krein) Psi route as a config field (audit 2026-10-16, F37).

``vfe3/geometry/phi_preconditioner.py`` documents ``"spectral"`` as an opt-in evaluation of the trivialized
exponential differentials Psi_R / Psi_L, but only the process-global ``set_phi_group_differential_mode``
reached it and no configuration could. ``m_phi_group_differential_mode`` ("series" | "spectral", default
"series") now selects it; ``GaugeManifoldAdamW`` takes it as ``phi_group_differential_mode`` (``None`` keeps the
process-global snapshot) without touching the process-global value, and checkpoints it as before.

Config level plus one tiny optimizer step; the launcher plumbing (``vfe3/train.py build_optimizer``) is lane C1's.
"""
from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

from tests.profiles import config_values, test_config
from vfe3.config import VFE3Config


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _pullback(**overrides) -> VFE3Config:
    return test_config(m_phi_update_mode="pullback_group", **overrides)


def test_default_is_the_series_route() -> None:
    assert VFE3Config.m_phi_group_differential_mode == "series"
    assert config_values()["m_phi_group_differential_mode"] == "series"


def test_spectral_is_accepted_and_unknown_values_refused() -> None:
    assert _pullback(m_phi_group_differential_mode="spectral").m_phi_group_differential_mode == "spectral"
    with pytest.raises(ValueError, match="m_phi_group_differential_mode"):
        _pullback(m_phi_group_differential_mode="daleckii")


@pytest.mark.parametrize("mode,inert", [("adamw", True), ("fisher_group", True), ("pullback_group", False)])
def test_marked_inert_off_the_pullback_group_route(mode: str, inert: bool) -> None:
    values = dict(m_phi_update_mode=mode)
    if mode == "fisher_group":
        values.update(gauge_group="block_glk", gauge_tying="per_head", encode_mode="per_token")
    try:
        cfg = test_config(**values)
    except ValueError as error:                       # a profile the fisher route refuses is not under test
        pytest.skip(f"fisher_group profile refused: {error}")
    assert ("m_phi_group_differential_mode" in cfg.inactive_fields()) is inert


def _optimizer(**kwargs):
    from vfe3.gauge_optim import GaugeManifoldAdamW
    from vfe3.geometry.groups import get_group

    group = get_group("block_glk")(4, 2)
    gen = torch.Generator().manual_seed(7)
    phi = nn.Parameter((0.2 * torch.randn(3, group.generators.shape[0], generator=gen)).float())
    optimizer = GaugeManifoldAdamW(
        [{"params": [phi], "lr": 0.05, "pullback_group": True, "weight_decay": 0.0}],
        group, phi_group_trust_radius=0.1, phi_chart_max_norm=5.0, weight_decay=0.0, **kwargs)
    phi.grad = torch.randn(phi.shape, generator=gen).float()
    return optimizer, phi


def test_optimizer_takes_the_mode_without_touching_the_process_global() -> None:
    from vfe3.geometry import phi_preconditioner as pp

    assert pp.phi_group_differential_mode() == "series"
    optimizer, _ = _optimizer(phi_group_differential_mode="spectral")
    assert optimizer._phi_group_differential_mode == "spectral"
    assert optimizer.state_dict()["phi_group_differential_mode"] == "spectral"
    assert pp.phi_group_differential_mode() == "series"
    snapshot, _ = _optimizer()                         # None keeps the historical process-global snapshot
    assert snapshot._phi_group_differential_mode == "series"
    with pytest.raises(ValueError, match="phi_group_differential_mode"):
        _optimizer(phi_group_differential_mode="daleckii")


def test_loaded_checkpoint_mode_replaces_the_constructed_one_with_a_warning() -> None:
    saved, _ = _optimizer(phi_group_differential_mode="spectral")
    state = saved.state_dict()
    resumed, _ = _optimizer(phi_group_differential_mode="series")
    with pytest.warns(UserWarning, match="replaces the constructed 'series'"):
        resumed.load_state_dict(state)
    assert resumed._phi_group_differential_mode == "spectral"
    same, _ = _optimizer(phi_group_differential_mode="spectral")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        same.load_state_dict(state)                    # matching modes load silently


@pytest.mark.parametrize("mode,expect_spectral", [("spectral", True), ("series", False)])
def test_optimizer_step_reaches_the_selected_route(monkeypatch, mode: str, expect_spectral: bool) -> None:
    from vfe3.geometry import phi_preconditioner as pp

    calls: list[int] = []
    original = pp._spectral_phi_differentials

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(pp, "_spectral_phi_differentials", counting)
    optimizer, phi = _optimizer(phi_group_differential_mode=mode)
    before = phi.detach().clone()
    optimizer.step()
    assert torch.isfinite(phi.detach()).all() and not torch.equal(phi.detach(), before)
    assert bool(calls) is expect_spectral
