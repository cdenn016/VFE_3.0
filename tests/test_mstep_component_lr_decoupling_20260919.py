"""Two M-step rates that were previously FORCED onto another component's rate.

``pos_phi_free`` (the learnable positional frame table, ``pos_phi='learned'``) rode ``m_phi_lr``
with ``phi_embed``, and the LayerNorm affine gamma/beta rode ``m_p_mu_lr`` with ``mu_embed`` and the
relational operator groups. Neither yoke followed from anything: the two frame tables already have
separate INIT scales (``phi_scale`` vs ``pos_phi_scale``) and differ in shape and row-visitation
statistics, while the affine is normalization calibration rather than prior-table capacity.

``m_pos_phi_lr`` and ``m_layernorm_lr`` are the new optional rates. Both default to ``None``, which
inherits the previous rate and reproduces the historical yoked behavior byte-for-byte, so every
existing config is unchanged. The yoke is kept where it IS motivated -- the two frame tables still
share ``phi_weight_decay``, because they compose into the ONE frame the transport clamp bounds.

These contracts pin the inheritance, the decoupled override, the auxiliary reporting role, the
inertness advisories, and the serialized-schema bump those two fields required.
"""

from tests.profiles import test_config

from typing import Any

import pytest
import torch

from vfe3.config import ConfigNotice, config_from_serialized, serialize_config
from vfe3.model.model import VFEModel
from vfe3.train import _learning_rates_by_aux_role, build_optimizer

from tests.test_relational_integration import _cfg

# Both seams live on this config: pos_phi='learned' creates pos_phi_free, and layernorm_affine=True
# on a 'layernorm' block seam creates the affine gamma/beta. That is also the shipped baseline in
# train_vfe3.py and ablation.py, so these tests exercise the real configuration, not a corner.
_LIVE = dict(use_head_mixer=False, use_block_mlp=False,
             pos_phi="learned", layernorm_affine=True,
             norm_type_block="layernorm", norm_type_final="layernorm", phi_weight_decay=0.065)

def _groups(**overrides: Any) -> tuple[Any, dict]:
    cfg = _cfg(**_LIVE, **overrides)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    name_of = {id(p): name for name, p in model.named_parameters()}
    optimizer = build_optimizer(model, cfg)
    by_param: dict = {}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            by_param[name_of[id(parameter)]] = group
    return optimizer, by_param


def test_both_rates_accept_the_inheriting_sentinel() -> None:
    """None selects inheritance independently of editable experiment defaults."""
    cfg = _cfg(m_pos_phi_lr=None, m_layernorm_lr=None)
    assert cfg.m_pos_phi_lr is None
    assert cfg.m_layernorm_lr is None


def test_pos_phi_free_inherits_m_phi_lr_when_unset() -> None:
    _, groups = _groups(m_phi_lr=0.0015, m_pos_phi_lr=None)
    group = groups["pos_phi_free"]
    assert group["lr"] == 0.0015
    assert group["role"] == "phi" and group["weight_decay"] == pytest.approx(0.065)


def test_pos_phi_free_takes_its_own_rate_when_set() -> None:
    _, groups = _groups(m_phi_lr=0.0015, m_pos_phi_lr=0.004)
    assert groups["pos_phi_free"]["lr"] == 0.004
    # The token frame is NOT dragged along by its sibling's override.
    assert groups["prior_bank.phi_embed"]["lr"] == 0.0015


def test_layernorm_affine_inherits_m_p_mu_lr_when_unset() -> None:
    _, groups = _groups(m_p_mu_lr=0.009, m_layernorm_lr=None)
    group = groups["block_norm.weight"]
    assert group["lr"] == 0.009
    assert group["weight_decay"] == 0.0            # calibration, not capacity


def test_layernorm_affine_takes_its_own_rate_when_set() -> None:
    _, groups = _groups(m_p_mu_lr=0.009, m_layernorm_lr=0.03)
    for name in ("block_norm.weight", "block_norm.bias", "final_norm.weight", "final_norm.bias"):
        assert groups[name]["lr"] == 0.03
    # mu_embed is NOT dragged along -- the whole point of the decoupling.
    assert groups["prior_bank.mu_embed"]["lr"] == 0.009


def test_layernorm_group_reports_its_own_auxiliary_rate() -> None:
    """The affine carries lr_aux_role='layernorm', so its scheduled rate reaches the CSV the way
    head_mixer/block_mlp do rather than being invisible."""
    optimizer, groups = _groups(m_p_mu_lr=0.009, m_layernorm_lr=0.03)
    assert groups["block_norm.weight"]["lr_aux_role"] == "layernorm"
    resolved = _learning_rates_by_aux_role(
        optimizer.param_groups, [g["lr"] for g in optimizer.param_groups])
    assert resolved["layernorm"] == 0.03


# ---------------------------------------------------------------------------
# inertness advisories: the new fields are inert when their target does not exist
# ---------------------------------------------------------------------------

_NOTICE_BASE = dict(vocab_size=8, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=1,
                    family="gaussian_diagonal", gauge_group="block_glk",
                    skip_belief_sigma_update=True, attention_mode="projected_gradient",
                    m_pos_phi_lr=None, m_layernorm_lr=None)


def _inert_fields(**overrides: Any) -> set:
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        test_config(**{**_NOTICE_BASE, **overrides})
    messages = [str(item.message) for item in caught
                if issubclass(item.category, ConfigNotice)]
    return {name for name in ("m_pos_phi_lr", "m_layernorm_lr")
            if any(name in message for message in messages)}


@pytest.mark.parametrize("overrides,expected", [
    (dict(pos_phi="learned", m_pos_phi_lr=0.004), set()),
    (dict(pos_phi="none", m_pos_phi_lr=0.004), {"m_pos_phi_lr"}),
    (dict(layernorm_affine=True, norm_type_block="layernorm", m_layernorm_lr=0.03), set()),
    (dict(layernorm_affine=False, m_layernorm_lr=0.03), {"m_layernorm_lr"}),
    # affine on, but neither seam is a 'layernorm' -> no gamma/beta are created at all
    (dict(layernorm_affine=True, norm_type_block="none", norm_type_final="none",
          m_layernorm_lr=0.03), {"m_layernorm_lr"}),
])
def test_inertness_advisories(overrides: dict, expected: set) -> None:
    assert _inert_fields(**overrides) == expected


def test_inheriting_component_rates_are_silent() -> None:
    """An abstaining config must not be nagged -- the advisory keys on a CHANGED value."""
    assert _inert_fields() == set()


# ---------------------------------------------------------------------------
# the serialized schema the two fields required
# ---------------------------------------------------------------------------

def test_the_component_rates_round_trip() -> None:
    cfg = _cfg(**_LIVE, m_pos_phi_lr=0.004, m_layernorm_lr=0.03)
    serialized = serialize_config(cfg)
    replayed = config_from_serialized(serialized, source="round-trip fixture")
    assert replayed.m_pos_phi_lr == 0.004
    assert replayed.m_layernorm_lr == 0.03


def test_an_unset_component_rate_inherits_its_component_step_size() -> None:
    """An absent component rate must resolve to None -- the component's own historical rate --
    rather than to any new step size."""
    replayed = _cfg(**_LIVE, m_pos_phi_lr=None, m_layernorm_lr=None)
    assert replayed.m_pos_phi_lr is None and replayed.m_layernorm_lr is None
    # ... and None is inheriting, so the groups land on the historical rates.
    torch.manual_seed(0)
    model = VFEModel(replayed)
    optimizer = build_optimizer(model, replayed)
    name_of = {id(p): name for name, p in model.named_parameters()}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if name_of[id(parameter)] == "pos_phi_free":
                assert group["lr"] == replayed.m_phi_lr


# ---------------------------------------------------------------------------
# the ablation sweeps registered for both
# ---------------------------------------------------------------------------

@pytest.fixture
def fixed_component_sweeps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test sweep behavior independently of the user's editable experiment choices."""
    import ablation
    baseline = serialize_config(_cfg(m_phi_lr=0.0015, m_p_mu_lr=0.009,
                                     m_pos_phi_lr=None, m_layernorm_lr=None))
    monkeypatch.setattr(ablation, "_resolved_sweep_baseline", lambda: dict(baseline))
    for name, values in (("m_pos_phi_lr", [0.0005, 0.0015, 0.003, 0.006]),
                         ("m_layernorm_lr", [0.001, 0.003, 0.009, 0.02])):
        monkeypatch.setitem(ablation.SWEEPS, name, dict(ablation.SWEEPS[name], values=values))


@pytest.mark.parametrize("name", ["m_pos_phi_lr", "m_layernorm_lr"])
def test_sweeps_are_registered_and_construct(name: str, fixed_component_sweeps: None) -> None:
    import ablation
    assert name in ablation.SWEEPS
    ablation.validate_sweeps([name], require_construction=True)
    assert ablation.sweep_n_runs(ablation.SWEEPS[name]) == 4


@pytest.mark.parametrize("name,inherited", [("m_pos_phi_lr", 0.0015), ("m_layernorm_lr", 0.009)])
def test_sweep_preserves_explicit_inherited_baseline_arm(
    name: str, inherited: float, fixed_component_sweeps: None,
) -> None:
    """A requested inherited-rate control survives arm generation and config resolution."""
    import ablation
    arms = ablation.make_run_overrides(name)
    controls = [overrides for _, overrides in arms if overrides[name] == inherited]
    assert len(controls) == 1
    cfg = test_config(**ablation._cell_cfg_dict(controls[0], seed=0))
    assert getattr(cfg, name) == inherited
    assert inherited == (cfg.m_phi_lr if name == "m_pos_phi_lr" else cfg.m_p_mu_lr)
