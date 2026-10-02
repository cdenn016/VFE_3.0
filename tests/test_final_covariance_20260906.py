"""Tiny final-loss covariance consumer and gradient regressions for S09."""
from tests.profiles import test_config
from collections.abc import Iterator
from dataclasses import replace
import importlib
import os

import pytest
import torch

from vfe3.families.gaussian import DiagonalGaussian
from vfe3.model.head_mixer import HeadMixer
from vfe3.model.model import VFEModel

block = importlib.import_module("vfe3.model.block")
prior_bank = importlib.import_module("vfe3.model.prior_bank")


@pytest.fixture(autouse=True)
def one_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def device() -> torch.device:
    return torch.device(os.environ.get("VFE3_TEST_DEVICE", "cpu"))


def _model(device: torch.device, **overrides: object) -> VFEModel:
    options = dict(vocab_size=7, embed_dim=4, n_heads=2, batch_size=1,
                   max_seq_len=4, n_layers=2, n_e_steps=1, decode_mode="linear",
                   use_head_mixer=True, use_block_mlp=True, block_mlp_expansion=2,
                   norm_type_block="none", norm_type_final="none",
                   amp_dtype=None)
    torch.manual_seed(27)
    model = VFEModel(test_config(**{**options, **overrides})).to(device)
    with torch.no_grad():
        for delta in model.head_mixer.mixer_deltas:
            delta.copy_(torch.randn_like(delta) * 0.07)
    return model


@pytest.mark.parametrize("layers", [1, 2])
@pytest.mark.parametrize("iterations", [1, 2])
@pytest.mark.parametrize("amp", [None, "bf16"])
def test_loss_and_all_named_gradients(layers: int, iterations: int, amp: str | None,
                                      device: torch.device, monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model(device, n_layers=layers, n_e_steps=iterations, amp_dtype=amp)
    tokens = torch.tensor([[0, 1, 0, 2]], device=device)
    targets = torch.tensor([[-100, 0, 2, 1]], device=device)
    actual = model(tokens, targets)[1]
    actual.backward()
    gradients = {name: None if p.grad is None else p.grad.clone() for name, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    monkeypatch.setattr(block, "_can_omit_mixer_covariance", lambda *args: False)
    expected = model(tokens, targets)[1]
    expected.backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for name, parameter in model.named_parameters():
        if gradients[name] is None or parameter.grad is None:
            assert gradients[name] is parameter.grad, name
        else:
            torch.testing.assert_close(gradients[name], parameter.grad, rtol=0, atol=0, msg=name)


@pytest.mark.parametrize("consumer", ["loss", "public", "capture", "logits", "block_norm", "final_norm",
    "mlp_delta", "custom_mlp", "custom_mixer", "mixer_hook", "global_hook", "decoder", "regularizer"])
def test_covariance_only_omitted_at_unused_final_seam(consumer: str, device: torch.device,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    overrides = {"mstep_self_coupling_weight": 0.1} if consumer == "regularizer" else {}
    model = _model(device, **overrides)
    tokens = torch.tensor([[0, 1, 0, 2]], device=device)
    calls = []
    original_mix = DiagonalGaussian.mix_dispersion
    def counted(sigma: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        calls.append(tuple(sigma.shape))
        return original_mix(sigma, matrix)
    monkeypatch.setattr(DiagonalGaussian, "mix_dispersion", staticmethod(counted))
    handles = []
    if consumer == "block_norm": model.block_norm = lambda mu, sigma: mu / sigma.sqrt()
    if consumer == "final_norm": model.final_norm = lambda mu, sigma: mu / sigma.sqrt()
    if consumer == "mlp_delta": model.block_mlps[-1].covariance_contract = "delta_full"
    if consumer == "custom_mlp":
        original = model.block_mlps[-1].forward_moments
        monkeypatch.setattr(model.block_mlps[-1], "forward_moments", lambda *args, **kwargs: original(*args, **kwargs))
    if consumer == "custom_mixer":
        original = model.head_mixer.forward
        monkeypatch.setattr(model.head_mixer, "forward", lambda *args, **kwargs: original(*args, **kwargs))
    if consumer == "mixer_hook": handles.append(model.head_mixer.register_forward_hook(lambda *args: None))
    if consumer == "global_hook":
        handles.append(torch.nn.modules.module.register_module_forward_hook(lambda *args: None))
    if consumer == "decoder":
        registration = prior_bank.get_decode_registration("linear")
        monkeypatch.setitem(prior_bank._DECODERS, "linear", replace(registration,
            fused_ce=lambda *args, **kwargs: registration.fused_ce(*args, **kwargs)))
    try:
        if consumer in ("public", "capture", "logits"):
            capture = {} if consumer == "capture" else None
            model.forward_beliefs(tokens, capture=capture, return_logits=consumer == "logits",
                                  _mean_only_loss=consumer != "public")
        else:
            model(tokens, torch.tensor([[1, 0, 2, 1]], device=device))[1].backward()
    finally:
        for handle in handles: handle.remove()
    assert len(calls) == (1 if consumer == "loss" else 2)


@pytest.mark.parametrize("full", [False, True])
def test_mixer_mean_gradients_hvp_and_finite_difference(full: bool, device: torch.device) -> None:
    torch.manual_seed(8)
    mixer = HeadMixer([2, 2]).to(device=device, dtype=torch.float64)
    with torch.no_grad(): mixer.mixer_deltas[0].normal_(std=0.1)
    mu = torch.randn(1, 3, 4, dtype=torch.float64, device=device, requires_grad=True)
    sigma = 1 + torch.rand_like(mu)
    if full: sigma = torch.diag_embed(sigma)
    leaves = (mu, *mixer.parameters())
    directions = tuple(torch.randn_like(p) for p in leaves)
    results = []
    for need_covariance in (True, False):
        mean, variance = mixer(mu, sigma, _need_covariance=need_covariance)
        if not need_covariance: assert variance is sigma
        loss = mean.square().sum()
        gradients = torch.autograd.grad(loss, leaves, create_graph=True)
        hvp = torch.autograd.grad(sum((g * d).sum() for g, d in zip(gradients, directions)), leaves)
        results.append((mean, *gradients, *hvp))
    for a, b in zip(*results): torch.testing.assert_close(a, b, rtol=0, atol=0)
    delta = 1e-5
    with torch.no_grad():
        plus = mixer(mu + delta * directions[0], sigma, _need_covariance=False)[0].square().sum()
        minus = mixer(mu - delta * directions[0], sigma, _need_covariance=False)[0].square().sum()
    torch.testing.assert_close((results[1][1] * directions[0]).sum(), (plus - minus) / (2 * delta),
                               rtol=1e-8, atol=1e-8)
