r"""Audit 2026-10-16 F18: the ``grow_K_mup`` scaling route applies the muP Adam learning-rate rule.

Yang et al. 2022 (Tensor Programs V, arXiv:2203.03466, Table 3) give the hidden and output matrices of a
width-:math:`K` network an Adam rate :math:`\propto 1/\mathrm{fan\_in} = 1/K` and keep the input table's rate and
init variance independent of width. The route's ``_mup`` arm therefore scales the relational score and content maps,
the block MLP and the linear readout by :math:`K_0/K`, and leaves ``m_p_mu_lr``, ``mu_init_std`` and the E-step rate
``e_q_mu_lr`` (an inference step size with no muP rule) at the baseline. The scaling guard refuses a cell only when
one of its overrides is inert under the cell's own config, so the route runs on a ``projected_predictive``
baseline. Config objects only; no model is built.
"""
from typing import Any, Dict, List

import pytest

import scaling
from tests.profiles import config_values
from vfe3.config import VFE3Config

_MUP_RATES = ("m_relational_score_lr", "m_relational_content_lr", "m_block_mlp_lr", "m_decode_mu_lr")


def _baseline(**overrides: Any) -> Dict[str, Any]:
    values = dict(vocab_size=7, embed_dim=4, n_heads=2, n_layers=1, n_e_steps=1, max_seq_len=3,
                  batch_size=2, max_steps=12, warmup_steps=1, eval_stride=None, gauge_group="block_glk",
                  pos_phi="none", attention_mode="projected_predictive", use_block_mlp=True,
                  block_mlp_expansion=2, kl_max=None, m_relational_score_lr=0.01,
                  m_relational_content_lr=0.0025, m_block_mlp_lr=0.001, m_decode_mu_lr=0.02)
    values.update(overrides)
    return config_values(**values)


@pytest.fixture(autouse=True)
def _local_scaling_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scaling, "CONFIG", dict(
        routes=["grow_K_mup"], dry_run=True, seeds=[1], data_seed=3, device="cpu", dataset="fixture",
        max_tokens=None, max_steps=None, resume=False, output_dir="unused-test-scaling-output"))


def _route(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> List[Dict[str, Any]]:
    monkeypatch.setattr(scaling, "BASELINE", _baseline(**overrides))
    cells = scaling.route_grow_k_mup([4, 8], n_heads=2, anchor_k=4)
    monkeypatch.setattr(scaling, "ROUTES", {"grow_K_mup": cells})
    return cells


def _cell(cells: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    return next(cell for cell in cells if cell["label"] == label)


def test_mup_arm_scales_hidden_and_output_rates_and_holds_the_input_table(monkeypatch: pytest.MonkeyPatch) -> None:
    cells = _route(monkeypatch)
    base = scaling.BASELINE
    common = {"embed_dim": 8, "n_heads": 2, "gauge_group": "block_glk"}
    assert _cell(cells, "K8_fixed")["overrides"] == common
    mup = _cell(cells, "K8_mup")["overrides"]
    assert set(mup) == set(common) | set(_MUP_RATES)
    for key in _MUP_RATES:
        assert mup[key] == pytest.approx(base[key] * 4 / 8)
    for key in ("e_q_mu_lr", "m_p_mu_lr", "mu_init_std"):
        assert key not in mup


def test_arms_coincide_at_the_anchor_width(monkeypatch: pytest.MonkeyPatch) -> None:
    cells = _route(monkeypatch)
    fixed = VFE3Config(**scaling._cell_cfg_dict(_cell(cells, "K4_fixed")["overrides"], 0, None))
    mup = VFE3Config(**scaling._cell_cfg_dict(_cell(cells, "K4_mup")["overrides"], 0, None))
    for key in _MUP_RATES + ("e_q_mu_lr", "m_p_mu_lr", "mu_init_std"):
        assert getattr(mup, key) == pytest.approx(getattr(fixed, key))


def test_route_is_admitted_on_a_projected_predictive_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    cells = _route(monkeypatch)
    scaling.validate_routes(["grow_K_mup"])
    for cell in cells:
        cfg = VFE3Config(**scaling._cell_cfg_dict(cell["overrides"], 0, None))
        assert cfg.relational_readout_mode == "predictive_content"
        assert not set(cell["overrides"]) & set(cfg.inactive_fields())


def test_inherited_baseline_rates_resolve_through_their_parent(monkeypatch: pytest.MonkeyPatch) -> None:
    cells = _route(monkeypatch, m_block_mlp_lr=None, m_decode_mu_lr=None, m_p_mu_lr=0.004)
    mup = _cell(cells, "K8_mup")["overrides"]
    assert mup["m_block_mlp_lr"] == pytest.approx(0.004 * 4 / 8)
    assert mup["m_decode_mu_lr"] == pytest.approx(0.004 * 4 / 8)


def test_guard_refuses_only_the_arm_whose_override_is_inert(monkeypatch: pytest.MonkeyPatch) -> None:
    cells = _route(monkeypatch, use_block_mlp=False)
    for label in ("K4_fixed", "K8_fixed"):
        cell = _cell(cells, label)
        scaling._validate_cell_controls(
            VFE3Config(**scaling._cell_cfg_dict(cell["overrides"], 0, None)), cell, None)
    cell = _cell(cells, "K8_mup")
    with pytest.raises(ValueError, match="m_block_mlp_lr"):
        scaling._validate_cell_controls(
            VFE3Config(**scaling._cell_cfg_dict(cell["overrides"], 0, None)), cell, None)
