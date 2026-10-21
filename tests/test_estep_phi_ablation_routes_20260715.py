"""Ablation sweep registry regressions.

The E-step-depth and phi-control arms this file was written for (estep_depth_damping,
phi_chart_control) were retired in the 2026-09-18 sweep cleanup, so the surviving regression here
uses an explicit local activation list and enabling profile.
"""

import pytest


def test_locally_selected_sweeps_validate(monkeypatch: pytest.MonkeyPatch) -> None:
    import ablation
    from tests.profiles import config_values

    monkeypatch.setattr(ablation, "BASELINE_CONFIG", config_values(use_block_mlp=True))
    monkeypatch.setattr(ablation, "fully_gauge_covariant", False)               # the driver switch is the user's; off here
    monkeypatch.setattr(ablation, "SWEEP_ORDER", ["block_mlp_expansion", "block_mlp_activation"])
    ablation.validate_sweeps(ablation.SWEEP_ORDER, require_construction=True)
