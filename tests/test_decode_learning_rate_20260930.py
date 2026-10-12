"""Decoder rate routing and owned CPU fused-target transfer regressions."""

from tests.profiles import test_config
from dataclasses import replace
from typing import Any

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel
from vfe3.model import prior_bank as prior_bank_module
from vfe3.model.prior_bank import get_decode_registration
from vfe3.train import _transfer_targets, build_optimizer


def _config(**overrides: Any) -> VFE3Config:
    values = dict(vocab_size=7, embed_dim=4, n_heads=2, max_seq_len=3,
                  n_layers=1, n_e_steps=1, use_block_mlp=False, pos_phi="none",
                  norm_type_block="none", norm_type_final="none", layernorm_affine=False,
                  decode_mode="diagonal_chunked")
    values.update(overrides)
    if values["decode_mode"] == "full_chunked":
        values["family"] = "gaussian_full"
    return test_config(**values)


def _groups(model: VFEModel, optimizer: torch.optim.Optimizer) -> dict[str, dict[str, Any]]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    members = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    assert len(members) == len({id(parameter) for parameter in members})
    assert {id(parameter) for parameter in members} == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    return {names[id(parameter)]: group for group in optimizer.param_groups
            for parameter in group["params"]}


@pytest.mark.parametrize("mode,untied", [("linear", False), ("diagonal_chunked", True)])
def test_default_preserves_historical_group_topology_and_parameter_bytes(
    mode:   str,
    untied: bool,
) -> None:
    options = dict(decode_mode=mode, untie_decode_bank=untied)
    torch.manual_seed(30)
    implicit = VFEModel(_config(**options))
    torch.manual_seed(30)
    explicit = VFEModel(_config(**options))
    optimizer = build_optimizer(implicit, implicit.cfg)
    explicit_optimizer = build_optimizer(explicit, explicit.cfg)
    _groups(implicit, optimizer)
    names = {id(parameter): name for name, parameter in implicit.named_parameters()}
    topology = [(tuple(names[id(parameter)] for parameter in group["params"]),
                 group["lr"], group["role"], group["weight_decay"], group.get("lr_report_role"))
                for group in optimizer.param_groups]
    expected = [(("prior_bank.mu_embed",), 0.025, "mu", 0.05, "mu"),
                (("prior_bank.sigma_log_embed",), 0.0025, "sigma", 0.05, "sigma"),
                (("prior_bank.phi_embed",), 0.015, "phi", 0.065, "phi")]
    expected += ([(("prior_bank.output_proj_weight",), 0.025, "mu", 0.05, None),
                  (("prior_bank.output_proj_bias",), 0.025, "mu", 0.0, None)]
                 if mode == "linear" else
                 [(("prior_bank.decode_mu_embed",), 0.025, "mu", 0.05, None),
                  (("prior_bank.decode_sigma_log_embed",), 0.0025, "sigma", 0.05, None)])
    expected += [(("relational_operators.raw_score_q", "relational_operators.raw_score_k"),
                  0.025, "mu", 0.0, None),
                 (("relational_operators.T", "relational_operators.O"),
                  0.025, "mu", 0.0, None)]
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


def test_decoder_override_keeps_temperature_freeze() -> None:
    cfg = _config(untie_decode_bank=True, m_decode_mu_lr=0.137, m_decode_sigma_lr=0.137)
    model = VFEModel(cfg)
    groups = _groups(model, build_optimizer(model, cfg))
    assert "prior_bank.decode_log_scale" not in groups


@pytest.mark.parametrize("mode", ["linear", "diagonal_chunked", "full_chunked",
                                  "expected_likelihood_chunked"])
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
                                  "expected_likelihood_chunked"])
def test_owned_cpu_targets_are_validated_before_any_builtin_fused_decoder(mode: str) -> None:
    model = VFEModel(_config(decode_mode=mode))
    with pytest.raises(IndexError, match="Target 7"):
        _transfer_targets(torch.tensor([[7, 1, 2]]), model, torch.device("cpu"))



@pytest.mark.parametrize("mode", ["linear", "diagonal_chunked", "full_chunked",
                                  "expected_likelihood_chunked"])
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
