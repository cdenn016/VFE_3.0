"""Independent decoder mean/covariance rates preserve encoder behavior.

The historical combined ``m_decode_lr`` is no longer a field; its load-time migration into these
split rates is pinned in tests/test_config_retired_fields_20261002.py.
"""

import inspect
from typing import Any, Optional

import pytest
import torch

from tests.profiles import config_values
from tests.test_decode_learning_rate_20260930 import _config, _groups
from vfe3.config import VFE3Config, config_from_serialized, serialize_config
from vfe3.model.model import VFEModel
from vfe3.train import _floor_lr_lambdas, build_optimizer


def test_independent_split_rates_keep_encoder_rates_and_leave_temperature_frozen() -> None:
    cfg = _config(untie_decode_bank=True, m_decode_mu_lr=0.137, m_decode_sigma_lr=0.061,
                  m_p_mu_lr=0.023, m_p_sigma_lr=0.004)
    model = VFEModel(cfg)
    groups = _groups(model, build_optimizer(model, cfg))
    assert groups["prior_bank.mu_embed"]["lr"] == 0.023
    assert groups["prior_bank.sigma_log_embed"]["lr"] == 0.004
    assert groups["prior_bank.decode_mu_embed"]["lr"] == 0.137
    assert groups["prior_bank.decode_sigma_log_embed"]["lr"] == 0.061
    assert "prior_bank.decode_log_scale" not in groups                # frozen, never grouped


@pytest.mark.parametrize("mu_rate,sigma_rate", [(None, None), (0.0, 0.0),
                                               (0.137, 0.061), (1, 2)])
def test_split_rates_roundtrip_and_omission(
    mu_rate:    Optional[float],
    sigma_rate: Optional[float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Omissions use a test-owned constructor baseline for this loader call.
    parameters = list(inspect.signature(VFE3Config.__init__).parameters)[1:]
    values = config_values()
    monkeypatch.setattr(VFE3Config.__init__, "__defaults__", tuple(values[name] for name in parameters))
    cfg = _config(m_decode_mu_lr=mu_rate, m_decode_sigma_lr=sigma_rate)
    saved = serialize_config(cfg)
    assert saved["m_decode_mu_lr"] == mu_rate
    assert saved["m_decode_sigma_lr"] == sigma_rate
    restored = config_from_serialized(saved, source="split decoder-rate test")
    assert restored.m_decode_mu_lr == mu_rate
    assert restored.m_decode_sigma_lr == sigma_rate
    saved.pop("m_decode_mu_lr")
    saved.pop("m_decode_sigma_lr")
    oldest = config_from_serialized(saved, source="historical encoder rates")
    assert oldest.m_decode_mu_lr is None
    assert oldest.m_decode_sigma_lr is None


@pytest.mark.parametrize("field", ["m_decode_mu_lr", "m_decode_sigma_lr"])
@pytest.mark.parametrize("rate", [-0.1, float("nan"), float("inf"), -float("inf"),
                                  True, False, "0.1", complex(0.1, 0.0)])
def test_split_rates_reject_invalid_explicit_values(field: str, rate: Any) -> None:
    with pytest.raises(ValueError, match=field):
        _config(**{field: rate})


@pytest.mark.parametrize("mode,untied", [("linear", False), ("diagonal_chunked", False),
                                         ("diagonal_chunked", True)])
def test_omitted_split_rates_preserve_group_topology_and_parameter_bytes(
    mode:        str,
    untied:      bool,
) -> None:
    options = dict(decode_mode=mode, untie_decode_bank=untied)
    torch.manual_seed(30)
    implicit = VFEModel(_config(**options))
    torch.manual_seed(30)
    explicit = VFEModel(_config(m_decode_mu_lr=None, m_decode_sigma_lr=None, **options))
    optimizer = build_optimizer(implicit, implicit.cfg)
    explicit_optimizer = build_optimizer(explicit, explicit.cfg)
    _groups(implicit, optimizer)
    _groups(explicit, explicit_optimizer)
    names = {id(parameter): name for name, parameter in implicit.named_parameters()}
    topology = [(tuple(names[id(parameter)] for parameter in group["params"]),
                 group["lr"], group["role"], group["weight_decay"], group.get("lr_report_role"))
                for group in optimizer.param_groups]
    expected = [(("prior_bank.mu_embed",), 0.025, "mu", 0.05, "mu"),
                (("prior_bank.sigma_log_embed",), 0.0025, "sigma", 0.05, "sigma"),
                (("prior_bank.phi_embed",), 0.015, "phi", 0.065, "phi")]
    if mode == "linear":
        expected += [(("prior_bank.output_proj_weight",), 0.025, "mu", 0.05, None),
                     (("prior_bank.output_proj_bias",), 0.025, "mu", 0.0, None)]
    elif untied:
        expected += [(("prior_bank.decode_mu_embed",), 0.025, "mu", 0.05, None),
                     (("prior_bank.decode_sigma_log_embed",), 0.0025, "sigma", 0.05, None)]
    expected += [(("relational_operators.raw_score_q", "relational_operators.raw_score_k"),
                  0.025, "mu", 0.0, None),
                 (("relational_operators.T", "relational_operators.O"),
                  0.025, "mu", 0.0, None)]
    assert topology == expected
    assert optimizer.state_dict() == explicit_optimizer.state_dict()
    for name, parameter in implicit.state_dict().items():
        assert torch.equal(parameter, explicit.state_dict()[name])
    for left, right in zip(implicit.parameters(), explicit.parameters()):
        if left.requires_grad:
            left.grad = torch.ones_like(left)
            right.grad = torch.ones_like(right)
    optimizer.step()
    explicit_optimizer.step()
    for name, parameter in implicit.state_dict().items():
        assert torch.equal(parameter, explicit.state_dict()[name])


@pytest.mark.parametrize("mu_rate,sigma_rate,want_mu,want_sigma", [
    (0.137, 0.061, 0.137, 0.061),
    (0.137, None, 0.137, 0.004),
    (None, 0.061, 0.023, 0.061),
    (None, None, 0.023, 0.004),
])
def test_untied_split_rates_follow_independent_precedence_and_preserve_other_roles(
    want_mu:          float,
    want_sigma:       float,
    mu_rate:          Optional[float],
    sigma_rate:       Optional[float],
) -> None:
    cfg = _config(untie_decode_bank=True, m_decode_mu_lr=mu_rate,
                  m_decode_sigma_lr=sigma_rate,
                  m_p_mu_lr=0.023, m_p_sigma_lr=0.004, m_phi_lr=0.013,
                  mu_weight_decay=0.07, sigma_weight_decay=0.09, phi_weight_decay=0.11)
    model = VFEModel(cfg)
    groups = _groups(model, build_optimizer(model, cfg))
    expected = {
        "prior_bank.mu_embed": (0.023, "mu", 0.07),
        "prior_bank.sigma_log_embed": (0.004, "sigma", 0.09),
        "prior_bank.phi_embed": (0.013, "phi", 0.11),
        "prior_bank.decode_mu_embed": (want_mu, "mu", 0.07),
        "prior_bank.decode_sigma_log_embed": (want_sigma, "sigma", 0.09),
    }
    for name, values in expected.items():
        group = groups[name]
        assert (group["lr"], group["role"], group["weight_decay"]) == values
    assert "prior_bank.decode_log_scale" not in groups                # frozen, never grouped


@pytest.mark.parametrize("mode", ["diagonal_chunked", "full_chunked"])
def test_tied_tables_keep_encoder_rates_when_both_split_overrides_are_explicit(mode: str) -> None:
    cfg = _config(decode_mode=mode, untie_decode_bank=False, m_decode_mu_lr=0.137,
                  m_decode_sigma_lr=0.061, m_p_mu_lr=0.023, m_p_sigma_lr=0.004)
    model = VFEModel(cfg)
    groups = _groups(model, build_optimizer(model, cfg))
    assert groups["prior_bank.mu_embed"]["lr"] == 0.023
    assert groups["prior_bank.sigma_log_embed"]["lr"] == 0.004
    assert "prior_bank.decode_log_scale" not in groups                # frozen, never grouped
    assert "prior_bank.decode_mu_embed" not in groups
    assert "prior_bank.decode_sigma_log_embed" not in groups


@pytest.mark.parametrize("mu_rate,want_mu", [(0.137, 0.137), (None, 0.023), (0.0, 0.0)])
def test_linear_readout_uses_mean_override_and_ignores_covariance_override(
    want_mu:          float,
    mu_rate:          Optional[float],
) -> None:
    cfg = _config(decode_mode="linear", m_decode_mu_lr=mu_rate, m_decode_sigma_lr=0.061,
                  m_p_mu_lr=0.023, m_p_sigma_lr=0.004,
                  mu_weight_decay=0.07, sigma_weight_decay=0.09)
    model = VFEModel(cfg)
    groups = _groups(model, build_optimizer(model, cfg))
    assert groups["prior_bank.mu_embed"]["lr"] == 0.023
    assert groups["prior_bank.sigma_log_embed"]["lr"] == 0.004
    assert "prior_bank.decode_log_scale" not in groups                # frozen, never grouped
    for name, decay in (("prior_bank.output_proj_weight", 0.07),
                        ("prior_bank.output_proj_bias", 0.0)):
        group = groups[name]
        assert (group["lr"], group["role"], group["weight_decay"]) == (want_mu, "mu", decay)


@pytest.mark.parametrize("field,frozen_name,live_name", [
    ("m_decode_mu_lr", "prior_bank.decode_mu_embed", "prior_bank.decode_sigma_log_embed"),
    ("m_decode_sigma_lr", "prior_bank.decode_sigma_log_embed", "prior_bank.decode_mu_embed"),
])
def test_explicit_zero_split_rate_stays_frozen_under_schedule(
    field:       str,
    frozen_name: str,
    live_name:   str,
) -> None:
    rates = {"m_decode_mu_lr": 0.137, "m_decode_sigma_lr": 0.061, field: 0.0}
    cfg = _config(untie_decode_bank=True, min_lr_frac=0.5,
                  warmup_steps=0, max_steps=4, **rates)
    model = VFEModel(cfg)
    optimizer = build_optimizer(model, cfg)
    groups = _groups(model, optimizer)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _floor_lr_lambdas([group["lr"] for group in optimizer.param_groups], cfg))
    initial = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    for _ in range(5):
        assert groups[frozen_name]["lr"] == 0.0
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        scheduler.step()
    parameters = dict(model.named_parameters())
    assert groups[frozen_name]["lr"] == 0.0
    assert torch.equal(parameters[frozen_name], initial[frozen_name])
    assert not torch.equal(parameters[live_name], initial[live_name])
    assert not torch.equal(parameters["prior_bank.mu_embed"], initial["prior_bank.mu_embed"])
    assert not torch.equal(parameters["prior_bank.sigma_log_embed"], initial["prior_bank.sigma_log_embed"])
    assert torch.equal(parameters["prior_bank.decode_log_scale"], initial["prior_bank.decode_log_scale"])


def test_split_rates_keep_frozen_temperature_out_of_optimizer() -> None:
    cfg = _config(untie_decode_bank=True, m_decode_mu_lr=0.137, m_decode_sigma_lr=0.061)
    model = VFEModel(cfg)
    groups = _groups(model, build_optimizer(model, cfg))
    assert groups["prior_bank.decode_mu_embed"]["lr"] == 0.137
    assert groups["prior_bank.decode_sigma_log_embed"]["lr"] == 0.061
    assert "prior_bank.decode_log_scale" not in groups


@pytest.mark.parametrize("mode,untied,want_inactive", [
    ("linear", False, {"m_decode_sigma_lr"}),
    ("diagonal_chunked", False, {"m_decode_mu_lr", "m_decode_sigma_lr"}),
    ("diagonal_chunked", True, set()),
])
def test_split_rate_activity_follows_independent_parameter_capacity(
    mode:          str,
    untied:        bool,
    want_inactive: set[str],
) -> None:
    cfg = _config(decode_mode=mode, untie_decode_bank=untied,
                  m_decode_mu_lr=0.137, m_decode_sigma_lr=0.061)
    assert cfg.inactive_fields().keys() & {"m_decode_mu_lr", "m_decode_sigma_lr"} == want_inactive
