"""Ablation sweep registry regressions.

The E-step-depth and phi-control arms this file was written for (estep_depth_damping,
phi_chart_control) were retired in the 2026-09-18 sweep cleanup, so the surviving regression here
is the activation invariant.
"""


def test_recommended_sweep_activation_matches_click_run_order() -> None:
    from ablation import SWEEPS, SWEEP_ORDER

    # Assert the INVARIANT, not the literal selection (audit 2026-07-25 F19). SWEEP_ORDER is the
    # owner's click-to-run choice of which sweeps to activate and changes constantly, so pinning its
    # contents made this test report a config edit as a regression. What must hold is that every
    # activated name resolves to a registered sweep and that the order names no duplicates.
    assert SWEEP_ORDER, "no sweeps activated"
    unknown = [name for name in SWEEP_ORDER if name not in SWEEPS]
    assert not unknown, f"SWEEP_ORDER names unregistered sweeps: {unknown}"
    assert len(set(SWEEP_ORDER)) == len(SWEEP_ORDER), "SWEEP_ORDER repeats a sweep"
