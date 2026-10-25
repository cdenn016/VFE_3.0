r"""The principled exact route in ablation.py: the ``principled_exact_route`` switch and the ``exact_route_*`` sweeps.

Every exact-route sweep must construct against BASELINE_CONFIG with the switch off (its ``requires`` carries the
route), every cell of a non-component sweep must meet the route, and the component sweep alone may leave it. With
the switch on, the resolved baseline is on the route, an arm that leaves the route is refused, and a sweep's base
cell is labeled at the resolved base value. Nothing is trained.
"""
import warnings

import pytest

import ablation
from vfe3.config import VFE3Config
from vfe3.exact_route import principled_exact_route_failures

EXACT_SWEEPS = sorted(name for name in ablation.SWEEPS if name.startswith("exact_route"))


def _cfg(sweep: str, overrides: dict) -> VFE3Config:
    cfg = ablation._resolved_sweep_baseline()
    cfg.update(overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VFE3Config(**cfg)


def test_exact_sweeps_construct_and_stay_on_the_route() -> None:
    assert "exact_route_components" in EXACT_SWEEPS and len(EXACT_SWEEPS) >= 8
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ablation.validate_sweeps(EXACT_SWEEPS)
    for name in EXACT_SWEEPS:
        for label, overrides in ablation.make_run_overrides(name):
            failures = principled_exact_route_failures(_cfg(name, overrides))
            if name == "exact_route_components" and label != "exact_route":
                assert failures, label                     # each component arm really leaves the route
            else:
                assert failures == [], (name, label, failures)


def test_component_arms_each_replace_a_distinct_component() -> None:
    runs = dict(ablation.make_run_overrides("exact_route_components"))
    requires = ablation.SWEEPS["exact_route_components"]["requires"]
    assert runs["exact_route"] == requires
    contrasts = {label: {k for k, v in ov.items() if requires.get(k, ablation._NO_SETTING) != v}
                 for label, ov in runs.items() if label != "exact_route"}
    assert all(contrasts.values())
    assert contrasts["decode_tau_0.1"] == {"decode_tau"}


def test_switch_resolves_baseline_and_refuses_off_route_arms(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ablation, "principled_exact_route", True)
    base = _cfg("", {})
    assert principled_exact_route_failures(base) == []
    ablation._require_exact_route(base, None)
    off = _cfg("", {"decode_tau": 0.1})
    with pytest.raises(ValueError, match="leaves the principled exact route"):
        ablation._require_exact_route(off, "decode_tau")
    ablation._require_exact_route(off, "exact_route_components")          # declared leaves_exact_route
    # The base cell of a param sweep sits at the resolved base value, not at BASELINE_CONFIG's literal.
    assert ablation._sweep_base_settings({})["decode_tau"] == 1.0
    labels = [label for label, _ in ablation.make_run_overrides("exact_route_regime_stay")]
    assert "block_memory_regime_stay=0.9" in labels


def test_switch_off_leaves_other_sweeps_alone() -> None:
    assert ablation.principled_exact_route is False
    ablation._require_exact_route(_cfg("", {}), "weight_decay")             # the plain baseline is not refused
    assert ablation._sweep_base_settings({})["block_mlp_mode"] == ablation.BASELINE_CONFIG["block_mlp_mode"]
