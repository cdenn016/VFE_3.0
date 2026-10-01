"""Decoder rate routing and owned CPU fused-target transfer regressions."""

from dataclasses import replace
from typing import Any, Optional

import pytest
import torch

from vfe3.config import VFE3Config, config_from_serialized, serialize_config
from vfe3.model.model import VFEModel
from vfe3.model import prior_bank as prior_bank_module
from vfe3.model.prior_bank import get_decode_registration
from vfe3.train import _floor_lr_lambdas, _transfer_targets, build_optimizer


def _config(**overrides: Any) -> VFE3Config:
    values = dict(vocab_size=7, embed_dim=4, n_heads=2, max_seq_len=3,
                  n_layers=1, n_e_steps=1, use_block_mlp=False, pos_phi="none",
                  norm_type_block="none", norm_type_final="none", layernorm_affine=False,
                  decode_mode="diagonal_chunked", decode_temperature_learned=True)
    values.update(overrides)
    if values["decode_mode"] == "full_chunked":
        values["family"] = "gaussian_full"
    return VFE3Config(**values)


def _groups(model: VFEModel, optimizer: torch.optim.Optimizer) -> dict[str, dict[str, Any]]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    members = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    assert len(members) == len({id(parameter) for parameter in members})
    assert {id(parameter) for parameter in members} == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    return {names[id(parameter)]: group for group in optimizer.param_groups
            for parameter in group["params"]}


@pytest.mark.parametrize("rate", [None, 0.0, 0.137])
def test_decoder_rate_serializes_and_reloads(rate: Optional[float]) -> None:
    cfg = _config(m_decode_lr=rate)
    saved = serialize_config(cfg)
    assert saved["m_decode_lr"] == rate
    assert config_from_serialized(saved, source="decoder-rate test").m_decode_lr == rate
    saved.pop("m_decode_lr")
    assert config_from_serialized(saved, source="historical config").m_decode_lr is None


@pytest.mark.parametrize("rate", [-0.1, float("nan"), float("inf"), -float("inf"), True, "0.1"])
def test_decoder_rate_rejects_invalid_explicit_values(rate: Any) -> None:
    with pytest.raises(ValueError, match="m_decode_lr"):
        _config(m_decode_lr=rate)


@pytest.mark.parametrize("mode,untied", [("linear", False), ("diagonal_chunked", True)])
def test_default_preserves_historical_group_topology_and_parameter_bytes(
    mode:   str,
    untied: bool,
) -> None:
    options = dict(decode_mode=mode, untie_decode_bank=untied)
    torch.manual_seed(30)
    implicit = VFEModel(_config(**options))
    torch.manual_seed(30)
    explicit = VFEModel(_config(m_decode_lr=None, **options))
    optimizer = build_optimizer(implicit, implicit.cfg)
    explicit_optimizer = build_optimizer(explicit, explicit.cfg)
    _groups(implicit, optimizer)
    names = {id(parameter): name for name, parameter in implicit.named_parameters()}
    topology = [(tuple(names[id(parameter)] for parameter in group["params"]),
                 group["lr"], group["role"], group["weight_decay"], group.get("lr_report_role"))
                for group in optimizer.param_groups]
    expected = [(("prior_bank.mu_embed",), 0.025, "mu", 0.05, "mu"),
                (("prior_bank.sigma_log_embed", "prior_bank.decode_log_scale"),
                 0.0025, "sigma", 0.05, "sigma"),
                (("prior_bank.phi_embed",), 0.015, "phi", 0.065, "phi")]
    expected += ([(("prior_bank.output_proj_weight",), 0.025, "mu", 0.05, None),
                  (("prior_bank.output_proj_bias",), 0.025, "mu", 0.0, None)]
                 if mode == "linear" else
                 [(("prior_bank.decode_mu_embed",), 0.025, "mu", 0.05, None),
                  (("prior_bank.decode_sigma_log_embed",), 0.0025, "sigma", 0.05, None)])
    assert topology == expected
    assert optimizer.state_dict() == explicit_optimizer.state_dict()
    for name, parameter in implicit.state_dict().items():
        assert torch.equal(parameter, explicit.state_dict()[name])
    # Same deterministic gradients give byte-identical default AdamW updates.
    for left, right in zip(implicit.parameters(), explicit.parameters()):
        if left.requires_grad:
            left.grad = torch.ones_like(left)
            right.grad = torch.ones_like(right)
    optimizer.step()
    explicit_optimizer.step()
    for name, parameter in implicit.state_dict().items():
        assert torch.equal(parameter, explicit.state_dict()[name])


@pytest.mark.parametrize("mode,untied", [("linear", False), ("diagonal_chunked", False),
                                         ("diagonal_chunked", True)])
@pytest.mark.parametrize("rate", [0.0, 0.137])
def test_explicit_decoder_rate_keeps_encoder_rates_roles_decays_and_exact_coverage(
    rate:   float,
    mode:   str,
    untied: bool,
) -> None:
    cfg = _config(decode_mode=mode, untie_decode_bank=untied, m_decode_lr=rate,
                  m_p_mu_lr=0.023, m_p_sigma_lr=0.004, mu_weight_decay=0.07,
                  sigma_weight_decay=0.09)
    model = VFEModel(cfg)
    groups = _groups(model, build_optimizer(model, cfg))
    assert groups["prior_bank.mu_embed"]["lr"] == 0.023
    assert groups["prior_bank.sigma_log_embed"]["lr"] == 0.004
    decoder_names = ["prior_bank.decode_log_scale"]
    if mode == "linear":
        decoder_names += ["prior_bank.output_proj_weight", "prior_bank.output_proj_bias"]
    elif untied:
        decoder_names += ["prior_bank.decode_mu_embed", "prior_bank.decode_sigma_log_embed"]
    for name in decoder_names:
        group = groups[name]
        assert group["lr"] == rate
        sigma = name in {"prior_bank.decode_log_scale", "prior_bank.decode_sigma_log_embed"}
        assert group["role"] == ("sigma" if sigma else "mu")
        expected_decay = 0.0 if name.endswith("output_proj_bias") else (0.09 if sigma else 0.07)
        assert group["weight_decay"] == expected_decay
        assert _floor_lr_lambdas([rate], cfg)[0](cfg.max_steps) * rate == pytest.approx(rate * cfg.min_lr_frac)


def test_zero_decoder_rate_freezes_separate_decode_but_not_encoder() -> None:
    cfg = _config(untie_decode_bank=True, m_decode_lr=0.0, min_lr=0.2)
    model = VFEModel(cfg)
    optimizer = build_optimizer(model, cfg)
    initial = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    for name in ("prior_bank.decode_mu_embed", "prior_bank.decode_sigma_log_embed",
                 "prior_bank.decode_log_scale"):
        assert torch.equal(dict(model.named_parameters())[name], initial[name])
    assert not torch.equal(model.prior_bank.mu_embed, initial["prior_bank.mu_embed"])
    assert _floor_lr_lambdas([0.0], cfg)[0](cfg.max_steps) * 0.0 == 0.0


def test_decoder_override_keeps_head_evidence_dedicated_rate_and_temperature_freeze() -> None:
    cfg = _config(untie_decode_bank=True, m_decode_lr=0.137, m_head_evidence_lr=0.081,
                  use_priorbank_head_evidence_mixer=True, decode_temperature_learned=False)
    model = VFEModel(cfg)
    groups = _groups(model, build_optimizer(model, cfg))
    assert groups["prior_bank.head_evidence_logits"]["lr"] == 0.081
    assert groups["prior_bank.head_evidence_logits"]["weight_decay"] == 0.0
    assert "prior_bank.decode_log_scale" not in groups


@pytest.mark.parametrize("mode", ["linear", "diagonal_chunked", "full_chunked",
                                  "expected_likelihood_chunked", "family_chunked"])
def test_owned_cpu_target_handoff_applies_to_each_builtin_fused_decoder(mode: str) -> None:
    model = VFEModel(_config(decode_mode=mode))
    original = torch.tensor([[1, -100, 2]])
    transferred = _transfer_targets(original, model, torch.device("cpu"))
    assert transferred is not original
    assert transferred.data_ptr() != original.data_ptr()
    assert torch.equal(transferred, original)
    assert transferred._vfe3_host_target_validation == (
        id(transferred), transferred._version, 7, -100, True)
    mu_q = torch.randn(1, 3, 4)
    sigma_q = (torch.diag_embed(torch.ones_like(mu_q)) if mode == "full_chunked"
               else torch.ones_like(mu_q))
    arguments = ((model.prior_bank, mu_q, transferred) if mode == "linear"
                 else (model.prior_bank, mu_q, sigma_q, transferred))
    result = get_decode_registration(mode).fused_ce(*arguments, return_stats=True)
    assert result.scored_tokens.item() == 2
    assert result.excluded_tokens.item() == 1
    assert torch.isfinite(result.ce)
    assert "_vfe3_host_target_validation" not in transferred.__dict__


@pytest.mark.parametrize("mode", ["linear", "diagonal_chunked", "full_chunked",
                                  "expected_likelihood_chunked", "family_chunked"])
def test_owned_cpu_targets_are_validated_before_any_builtin_fused_decoder(mode: str) -> None:
    model = VFEModel(_config(decode_mode=mode))
    with pytest.raises(IndexError, match="Target 7"):
        _transfer_targets(torch.tensor([[7, 1, 2]]), model, torch.device("cpu"))



@pytest.mark.parametrize("mode", ["linear", "diagonal_chunked", "full_chunked",
                                  "expected_likelihood_chunked", "family_chunked"])
def test_replaced_fused_callable_keeps_normal_target_validation(
    mode:        str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = VFEModel(_config(decode_mode=mode))
    original_registration = get_decode_registration(mode)

    def custom_ce(*args: Any, **kwargs: Any) -> Any:
        return original_registration.fused_ce(*args, **kwargs)

    monkeypatch.setitem(prior_bank_module._DECODERS, mode,
                        replace(original_registration, fused_ce=custom_ce))
    targets = torch.tensor([[1, 2, 3]])
    transferred = _transfer_targets(targets, model, torch.device("cpu"))
    assert transferred is targets
    assert "_vfe3_host_target_validation" not in transferred.__dict__
