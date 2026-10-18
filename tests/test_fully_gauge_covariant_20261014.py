r"""The ``fully_gauge_covariant`` launcher switch (:mod:`vfe3.gauge_covariant`).

Config-level only (no model is built). Pinned:

* the overrides take non-covariant launcher configurations onto the gauge facet of the pure-path report (every
  ``gauge_flags`` entry true), and the unmodified configurations are not on it, so the check discriminates;
* the content links (exponential chart, low-rank Lie-algebra edges) and the other non-gauge keys are never touched;
* an already covariant configuration gets no overrides;
* combinations with no covariant reading that keeps the chosen block are refused.
"""
import warnings

import pytest

from tests.profiles import config_values
from vfe3.config import VFE3Config
from vfe3.gauge_covariant import fully_gauge_covariant_overrides, gauge_covariance_failures

# Keys that are not gauge seams; the switch must leave them alone.
UNTOUCHED = ("content_link_chart", "content_link_edges", "content_link_edge_rank", "content_link_latent",
             "relational_routing", "predictive_readout", "predictive_covariance", "attention_mode",
             "beta_attention_prior", "use_block_mlp", "block_mlp_mode", "norm_type_block", "gauge_group",
             "m_phi_lr", "embed_dim", "n_heads")


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _base(**overrides) -> dict:
    r"""Projected-predictive kernel readout with the frame-dependent defaults of the test profile (diagonal family,
    local block and decoder, ``sigma_max``, per-coordinate self term)."""
    return config_values(attention_mode="projected_predictive", predictive_readout="kernel",
                         predictive_covariance="moment_matched", lambda_alpha_mode="state_dependent_per_coord",
                         **overrides)


CASES = {
    "diagonal_linear":        _base(),
    "diagonal_chunked":       _base(decode_mode="diagonal_chunked"),
    "full_local":             _base(family="gaussian_full", decode_mode="full_chunked"),
    "exponential_links":      _base(content_link_chart="exponential", decode_mode="diagonal_chunked"),
    "lowrank_edges":          _base(family="gaussian_full", decode_mode="full_chunked",
                                    content_link_chart="exponential", content_link_edges="lowrank"),
    "comparison_rope":        _base(rope_mode="comparison_invariant"),
    "frame_rope":             _base(rope_mode="frame"),
    "pullback_phi":           _base(m_phi_update_mode="pullback_group"),
    "no_block_mlp":           _base(use_block_mlp=False),
    "gauge_gate_mlp":         _base(block_mlp_mode="gauge_gate", norm_type_block="none"),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_overrides_reach_the_gauge_facet_and_touch_nothing_else(case: str) -> None:
    config = CASES[case]
    before = VFE3Config(**config)
    overrides = fully_gauge_covariant_overrides(config)
    after = VFE3Config(**{**config, **overrides})
    assert gauge_covariance_failures(before)                 # the unmodified configuration is not covariant
    assert gauge_covariance_failures(after) == []
    for key in UNTOUCHED:
        assert key not in overrides
        assert getattr(after, key) == getattr(before, key)


def test_a_covariant_configuration_gets_no_overrides() -> None:
    config = CASES["lowrank_edges"]
    covariant = {**config, **fully_gauge_covariant_overrides(config)}
    assert fully_gauge_covariant_overrides(covariant) == {}


@pytest.mark.parametrize("config, match", [
    (_base(block_mlp_mode="gauge_gate", norm_type_block="layernorm"), "covariant only in local coordinates"),
    (_base(norm_type_block="mahalanobis"), "admits norm_type_block"),
])
def test_combinations_without_a_covariant_reading_are_refused(config: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        fully_gauge_covariant_overrides(config)


def test_launcher_switch_is_a_bool() -> None:
    r"""The switch's value is the user's; only its validity is asserted (audit 2026-10-17, item 38)."""
    import ablation
    import train_vfe3

    assert isinstance(train_vfe3.fully_gauge_covariant, bool)
    assert isinstance(ablation.fully_gauge_covariant, bool)


def test_ablation_switch_overrides_the_baseline_and_refuses_breaking_arms(monkeypatch: pytest.MonkeyPatch) -> None:
    import ablation

    monkeypatch.setattr(ablation, "fully_gauge_covariant", False)
    assert ablation._resolved_sweep_baseline() == ablation.BASELINE_CONFIG          # off: the baseline as written
    config = CASES["diagonal_chunked"]
    monkeypatch.setattr(ablation, "BASELINE_CONFIG", config)
    monkeypatch.setattr(ablation, "fully_gauge_covariant", True)
    baseline = ablation._resolved_sweep_baseline()
    assert baseline == {**config, **fully_gauge_covariant_overrides(config)}
    ablation._require_gauge_covariant(VFE3Config(**baseline))                      # the covariant baseline passes
    with pytest.raises(ValueError, match="breaks gauge covariance"):                 # an arm that re-breaks it
        ablation._require_gauge_covariant(VFE3Config(**{**baseline, "block_coordinates": "local"}))
    cell = ablation._cell_cfg_dict({"m_phi_lr": 0.002}, seed=0)                     # cells read the switched baseline
    assert cell["family"] == "gaussian_full" and cell["m_phi_lr"] == 0.002
