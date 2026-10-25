r"""Tests for ``block_mlp_tie_layers``: one block MLP shared by every layer.

A tied model's state dict loads into an untied model with the same configuration, giving every layer the same bank.
The two models then compute the same loss, and the tied gradient of each shared parameter equals the sum over layers
of the untied gradients: one set of dynamics applied at every depth. The tests also cover the module topology and
parameter counts, optimizer groups without duplicates, the load guard against untied checkpoints, one calibration of
the shared bank, and the configuration field. CPU only, one thread, tiny models.
"""

from __future__ import annotations

import warnings
from typing import Dict

import pytest
import torch

from tests.profiles import test_config
from vfe3.model.model import VFEModel
from vfe3.train import build_optimizer

torch.set_num_threads(1)

_COORDINATE = dict(attention_mode="projected_predictive", block_mlp_mode="coordinate")
_PROTOTYPE = dict(attention_mode="projected_predictive", block_mlp_mode="prototype_agents")
_MEMORY = dict(attention_mode="projected_predictive", predictive_readout="minimal_agent",
               minimal_agent_inference="exact", family="gaussian_full", decode_mode="full_chunked",
               norm_type_block="none", block_coordinates="reference", block_mlp_mode="cue_result_memory",
               block_mlp_inference="exact", block_mlp_covariance="posterior", block_memory_regression_rank=2)
ROUTES: Dict[str, dict] = {"coordinate": _COORDINATE, "prototype_agents": _PROTOTYPE, "cue_result_memory": _MEMORY}


def _cfg(route: str, **overrides):
    values = dict(ROUTES[route], use_block_mlp=True, block_mlp_expansion=2, n_layers=3, max_seq_len=4,
                  block_mlp_dropout=0.0)
    values.update(overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return test_config(**values)


def _model(cfg, seed: int = 0) -> VFEModel:
    torch.manual_seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VFEModel(cfg)


def _loss(model: VFEModel, tokens: torch.Tensor) -> torch.Tensor:
    model.train()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, loss, _ = model(tokens, tokens)
    return loss


@pytest.mark.parametrize("route", sorted(ROUTES))
def test_tied_topology_counts_and_state_dict(route: str) -> None:
    tied, untied = _model(_cfg(route, block_mlp_tie_layers=True)), _model(_cfg(route))
    assert all(m is tied.block_mlps[0] for m in tied.block_mlps)
    assert len({id(m) for m in untied.block_mlps}) == untied.cfg.n_layers
    per_layer = sum(p.numel() for p in untied.block_mlps[0].parameters())
    assert sum(p.numel() for p in tied.block_mlps.parameters()) == per_layer
    assert tied.executable_build.block_mlp.parameter_count == per_layer
    assert untied.executable_build.block_mlp.parameter_count == per_layer * untied.cfg.n_layers
    assert tied.executable_build.block_mlp.flops_per_token == untied.executable_build.block_mlp.flops_per_token
    keys = sorted(k for k in tied.state_dict() if k.startswith("block_mlps."))
    assert keys == sorted(k for k in untied.state_dict() if k.startswith("block_mlps."))


@pytest.mark.parametrize("route", sorted(ROUTES))
def test_tied_gradient_is_the_sum_of_per_layer_gradients(route: str) -> None:
    tied = _model(_cfg(route, block_mlp_tie_layers=True), seed=1)
    with torch.no_grad():                                    # move off any symmetric initialization
        for p in tied.block_mlps[0].parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=torch.Generator().manual_seed(p.numel()), dtype=p.dtype))
    untied = _model(_cfg(route), seed=2)
    untied.load_state_dict(tied.state_dict())                # every layer receives the shared bank
    tokens = torch.randint(0, tied.cfg.vocab_size, (2, 4), generator=torch.Generator().manual_seed(3))
    torch.manual_seed(4)
    loss_tied = _loss(tied, tokens)
    torch.manual_seed(4)
    loss_untied = _loss(untied, tokens)
    assert torch.allclose(loss_tied, loss_untied, rtol=1e-6, atol=1e-7)
    loss_tied.backward()
    loss_untied.backward()
    for name, p in tied.block_mlps[0].named_parameters():
        summed = sum(dict(mlp.named_parameters())[name].grad for mlp in untied.block_mlps)
        assert p.grad is not None
        assert torch.allclose(p.grad, summed, rtol=1e-5, atol=1e-7), name


@pytest.mark.parametrize("route", sorted(ROUTES))
def test_optimizer_lists_each_shared_parameter_once(route: str) -> None:
    model = _model(_cfg(route, block_mlp_tie_layers=True))
    optimizer = build_optimizer(model, model.cfg)
    seen = [id(p) for g in optimizer.param_groups for p in g["params"]]
    assert len(seen) == len(set(seen))
    assert {id(p) for p in model.block_mlps.parameters()} <= set(seen)


def test_load_guard_refuses_an_untied_checkpoint_and_round_trips() -> None:
    cfg_tied = _cfg("prototype_agents", block_mlp_tie_layers=True)
    untied = _model(_cfg("prototype_agents"), seed=5)
    tied = _model(cfg_tied, seed=6)
    with pytest.raises(ValueError, match="untied"):
        tied.load_state_dict(untied.state_dict())
    restored = _model(cfg_tied, seed=7)
    restored.load_state_dict(tied.state_dict())
    for a, b in zip(restored.block_mlps[0].parameters(), tied.block_mlps[0].parameters()):
        assert torch.equal(a, b)
    assert all(m is restored.block_mlps[0] for m in restored.block_mlps)


def test_working_variance_calibration_runs_once_on_the_shared_bank() -> None:
    model = _model(_cfg("prototype_agents", block_mlp_tie_layers=True, block_mlp_template_init="working_variance"))
    tokens = torch.randint(0, model.cfg.vocab_size, (3, 4), generator=torch.Generator().manual_seed(8))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stats = model.calibrate_prototype_templates(tokens, pull=0.2, seed=9)
    assert len(stats) == 1


def test_configuration_field() -> None:
    assert _cfg("coordinate").block_mlp_tie_layers is False
    for bad in (1, "yes", None):
        with pytest.raises(ValueError, match="block_mlp_tie_layers"):
            _cfg("coordinate", block_mlp_tie_layers=bad)
    assert "block_mlp_tie_layers" not in set(_cfg("coordinate", block_mlp_tie_layers=True).inactive_fields())
    assert "block_mlp_tie_layers" in set(_cfg("coordinate", use_block_mlp=False).inactive_fields())
    assert "block_mlp_tie_layers" in set(_cfg("coordinate", n_layers=1).inactive_fields())
