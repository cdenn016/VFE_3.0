"""Cross-cutting checks on the live launcher route at tiny scale (``tests.profiles.live_config``).

The default test profile is the legacy projected_gradient route; these checks run the route
``train_vfe3.py`` trains: projected_predictive with the kernel readout and a moment-matched covariance,
a layernorm block norm, the coordinate block MLP, the causal ALiBi prior, the row_min routing clamp and a
linear decode, stepped by AdamW.
"""
from dataclasses import asdict, fields

import torch

from tests.profiles import config_values, live_config, live_values
from vfe3.config import VFE3Config, config_from_serialized, serialize_config
from vfe3.model.model import VFEModel
from vfe3.train import build_optimizer


def _model(seed: int = 0) -> VFEModel:
    torch.manual_seed(seed)
    return VFEModel(live_config())


def _batch(cfg: VFE3Config, seed: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    shape = (cfg.batch_size, cfg.max_seq_len)
    return (torch.randint(0, cfg.vocab_size, shape, generator=g),
            torch.randint(0, cfg.vocab_size, shape, generator=g))


def test_live_profile_is_the_launcher_route() -> None:
    cfg = live_config()
    assert (cfg.attention_mode, cfg.predictive_readout, cfg.predictive_covariance) == (
        "projected_predictive", "kernel", "moment_matched")
    assert (cfg.family, cfg.gauge_group, cfg.norm_type_block, cfg.block_mlp_mode) == (
        "gaussian_diagonal", "block_glk", "layernorm", "coordinate")
    assert (cfg.beta_attention_prior, cfg.routing_kl_max_reference, cfg.decode_mode, cfg.m_phi_update_mode) == (
        "causal_alibi", "row_min", "linear", "adamw")
    assert cfg.use_block_mlp and cfg.kl_max == 8.0 * cfg.embed_dim


def test_live_forward_backward_is_finite() -> None:
    model = _model()
    tokens, targets = _batch(model.cfg)
    _, loss, _ = model(tokens, targets)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(bool(torch.isfinite(g).all()) for g in grads)
    assert any(bool((g != 0).any()) for g in grads)


def test_live_route_is_causal() -> None:
    model = _model().eval()
    tokens, _ = _batch(model.cfg)
    j = 2
    perturbed = tokens.clone()
    perturbed[:, j] = (perturbed[:, j] + 1) % model.cfg.vocab_size
    with torch.no_grad():
        logits = model(tokens)                                   # (B, N, V)
        changed = model(perturbed)
    assert torch.equal(logits[:, :j], changed[:, :j])            # earlier positions bitwise unchanged
    assert not torch.equal(logits[:, j:], changed[:, j:])        # the perturbation does reach position j


def test_live_optimizer_step_moves_parameters_and_stays_finite() -> None:
    model = _model()
    optimizer = build_optimizer(model, model.cfg)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    tokens, targets = _batch(model.cfg)
    _, loss, _ = model(tokens, targets)
    loss.backward()
    optimizer.step()
    after = dict(model.named_parameters())
    assert all(bool(torch.isfinite(p).all()) for p in after.values())
    assert any(not torch.equal(before[name], p.detach()) for name, p in after.items())
    _, loss_after, _ = model(tokens, targets)
    assert torch.isfinite(loss_after)


def test_live_profile_is_isolated_and_complete() -> None:
    assert {field.name for field in fields(VFE3Config)} == set(live_values())
    first = live_values()
    first["family"] = "gaussian_full"
    assert live_values()["family"] == "gaussian_diagonal"
    assert config_values()["attention_mode"] == "projected_gradient"     # the legacy baseline is unchanged
    cfg = live_config()
    assert config_from_serialized(serialize_config(cfg), source="live profile") == cfg
    assert asdict(live_config(n_layers=1))["n_layers"] == 1 and live_config().n_layers == 2
