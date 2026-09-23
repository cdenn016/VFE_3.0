"""Tiny executable routing contracts for the optional full-Gaussian scorer."""

import importlib
from typing import Any

import pytest
import torch

from vfe3.belief import BeliefState
from vfe3.config import VFE3Config, config_from_serialized, serialize_config
from vfe3.families.gaussian import FullGaussian
from vfe3.model.block import e_step_shared_kwargs
from vfe3.model.model import VFEModel


def _tiny_config(**overrides: Any) -> VFE3Config:
    values = dict(
        vocab_size=5, embed_dim=4, n_heads=2, max_seq_len=2,
        batch_size=1, n_layers=1, n_e_steps=1,
        family="gaussian_full", gauge_group="block_glk", transport_mode="flat",
        cross_couplings=None, pos_phi="none", pos_rotation="none",
        e_step_update="gradient", e_step_gradient="unroll",
        oracle_unroll_grad=True, compile_pair_kernel=False,
        e_phi_lr=0.005, phi_precond_mode="killing_per_block",
        decode_mode="linear", mu_init_std=0.01, phi_scale=0.01,
    )
    values.update(overrides)
    return VFE3Config(**values)


def test_backend_default_is_common_frame() -> None:
    # The default moved from "pairwise" to "common_frame" in f54c756; the round-trip and opt-in
    # coverage below is what makes either value safe to reach.
    cfg = _tiny_config()
    assert cfg.full_cov_kl_backend == "common_frame"
    assert e_step_shared_kwargs(cfg, torch.device("cpu"))["full_cov_kl_backend"] == "common_frame"


def test_backend_round_trip_and_shared_kwargs_preserve_opt_in() -> None:
    cfg = _tiny_config(full_cov_kl_backend="common_frame")
    restored = config_from_serialized(serialize_config(cfg), source="backend routing test")
    assert restored.full_cov_kl_backend == "common_frame"
    assert e_step_shared_kwargs(restored, torch.device("cpu"))["full_cov_kl_backend"] == "common_frame"


@pytest.mark.parametrize("value", ["automatic", None, 1])
def test_backend_rejects_unknown_values(value: object) -> None:
    with pytest.raises(ValueError, match="full_cov_kl_backend"):
        _tiny_config(full_cov_kl_backend=value)


@pytest.mark.parametrize("module_name", ["vfe3.inference.e_step", "vfe3.gradients.oracle"])
def test_custom_family_does_not_receive_builtin_score_keyword(module_name: str) -> None:
    module = importlib.import_module(module_name)

    class CustomFamily:
        @classmethod
        def coupling_energy(cls, value: torch.Tensor) -> torch.Tensor:
            return value

    value = torch.tensor(3.0)
    assert module._coupling_energy(CustomFamily, value, score_backend="common_frame") is value


def test_model_policy_is_owned_and_custom_family_api_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    first = VFEModel(_tiny_config(full_cov_kl_backend="common_frame"))
    # Explicit opt-out: the default is common_frame now, so discriminating "the model owns the
    # policy" needs two models that genuinely carry different backends.
    second = VFEModel(_tiny_config(full_cov_kl_backend="pairwise"))
    seen = []

    def capture(cls: type, *args: object, **kwargs: object) -> torch.Tensor:
        seen.append(kwargs["score_backend"])
        return torch.tensor(1.0)

    monkeypatch.setattr(FullGaussian, "coupling_energy", classmethod(capture))
    first._coupling_energy(FullGaussian)
    second._coupling_energy(FullGaussian)
    first._coupling_energy(FullGaussian)
    assert seen == ["common_frame", "pairwise", "common_frame"]

    class CustomFamily:
        @classmethod
        def coupling_energy(cls, value: torch.Tensor) -> torch.Tensor:
            return value

    value = torch.tensor(2.0)
    assert first._coupling_energy(CustomFamily, value) is value


@pytest.mark.parametrize("update", ["gradient", "mm_exact"])
def test_tiny_forward_and_backward_forward_backend_to_coupling(
    monkeypatch: pytest.MonkeyPatch,
    update: str,
) -> None:
    cfg = _tiny_config(full_cov_kl_backend="common_frame", e_step_update=update)
    model = VFEModel(cfg)
    original = FullGaussian.coupling_energy
    seen = []

    def capture(cls: type, *args: object, **kwargs: object) -> torch.Tensor:
        seen.append(kwargs.get("score_backend"))
        return original(*args, **kwargs)

    monkeypatch.setattr(FullGaussian, "coupling_energy", classmethod(capture))
    logits = model(torch.tensor([[0, 1]]))
    logits.square().sum().backward()
    e_step_module = importlib.import_module("vfe3.inference.e_step")
    belief = BeliefState(
        mu=torch.zeros(2, cfg.embed_dim),
        sigma=torch.eye(cfg.embed_dim).expand(2, -1, -1).clone(),
        phi=torch.zeros(2, model.group.generators.shape[0]),
    )
    diagnostic = e_step_module.free_energy_value(
        belief, belief.mu, belief.sigma, model.group,
        **e_step_shared_kwargs(cfg, torch.device("cpu")),
    )
    assert torch.isfinite(diagnostic)
    assert seen and all(backend == "common_frame" for backend in seen)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
