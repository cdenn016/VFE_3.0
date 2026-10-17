"""Cross-cutting checks on the live launcher route at tiny scale (``tests.profiles.live_config``).

The default test profile is the legacy projected_gradient route; these checks run the route
``train_vfe3.py`` trains. The profile takes every key of the launcher's ``config`` dictionary, shrinks the
shape and run fields to a CPU unit test, and pins the fields the launcher omits at their defaults, so a
route switch in the launcher reaches these checks without an edit here.
"""
import dataclasses
from dataclasses import asdict, fields

import torch

from tests.profiles import _LIVE_TINY, config_values, launcher_values, live_config, live_values
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
    launcher = launcher_values()
    values = asdict(live_config())
    resized = set(_LIVE_TINY)
    # Only shape, precision and run fields are resized; every route switch is the launcher's.
    assert resized == {"vocab_size", "embed_dim", "n_heads", "max_seq_len", "eval_stride", "batch_size",
                       "n_layers", "max_steps", "warmup_steps", "block_mlp_expansion", "kl_max", "amp_dtype",
                       "log_interval", "eval_interval", "checkpoint_interval", "generate_embedding_figures"}
    followed = {name: (values[name], launcher[name]) for name in launcher if name not in resized}
    assert {name: pair for name, pair in followed.items() if pair[0] != pair[1]} == {}
    defaults = {field.name: field.default for field in fields(VFE3Config)
                if field.default is not dataclasses.MISSING}
    omitted = {name for name in values if name not in launcher and name not in resized}
    assert {name: (values[name], defaults[name]) for name in omitted if values[name] != defaults[name]} == {}
    assert live_config().kl_max == 8.0 * live_config().embed_dim


def test_live_forward_backward_is_finite() -> None:
    model = _model()
    tokens, targets = _batch(model.cfg)
    _, loss, _ = model(tokens, targets)
    assert torch.isfinite(loss)
    loss.backward()
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    # Every trainable parameter is reached by backward (audit 2026-10-16 F111): a dead table or a severed
    # detach leaves its gradient None.
    assert trainable and [name for name, p in trainable.items() if p.grad is None] == []
    grads = [p.grad for p in trainable.values()]
    assert all(bool(torch.isfinite(g).all()) for g in grads)
    assert any(bool((g != 0).any()) for g in grads)


def test_live_backward_matches_float64_central_differences() -> None:
    r"""Each trainable parameter's gradient on the live route matches a float64 central difference (audit
    2026-10-16 F111).

    For one random direction :math:`d_p` per parameter :math:`p`,
    :math:`(L(p + h d_p) - L(p - h d_p))/(2h) = \langle\nabla_p L, d_p\rangle + O(h^2)`. A wrong custom
    backward, a missing chain-rule term or a stray detach on the launcher route moves the analytic side away
    from the difference. The block-norm gains start tied at one; they are moved off the tie so that no
    maximum over the gains is differentiated at a kink.
    """
    torch.manual_seed(0)
    model = VFEModel(live_config()).double()
    with torch.no_grad():
        generator = torch.Generator().manual_seed(5)
        for name, p in model.named_parameters():
            if name.endswith("norm.weight"):
                p.add_(0.01 * torch.randn(p.shape, dtype=p.dtype, generator=generator))
    tokens, targets = _batch(model.cfg)

    def loss() -> torch.Tensor:
        return model(tokens, targets)[1]

    loss().backward()
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    assert trainable and [name for name, p in trainable.items() if p.grad is None] == []
    h = 1e-5
    directions = torch.Generator().manual_seed(2)
    mismatched = {}
    for name, p in trainable.items():
        d = torch.randn(p.shape, dtype=p.dtype, generator=directions)
        original = p.detach().clone()
        with torch.no_grad():
            p.copy_(original + h * d)
            plus = float(loss())
            p.copy_(original - h * d)
            minus = float(loss())
            p.copy_(original)
        difference = (plus - minus) / (2.0 * h)
        analytic = float((p.grad * d).sum())
        if abs(difference - analytic) > 1e-5 * max(abs(difference), abs(analytic)) + 1e-8:
            mismatched[name] = (difference, analytic)
    assert mismatched == {}


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
    scored = torch.tensor(targets.numel(), dtype=torch.int64)
    for name in ("phi_statistics_provider", "table_statistics_provider"):   # the Fisher M-steps, as train_step
        provider = getattr(optimizer, name, None)
        if provider is not None:
            provider.set_batch(tokens, scored)
    optimizer.step()
    after = dict(model.named_parameters())
    assert all(bool(torch.isfinite(p).all()) for p in after.values())
    assert any(not torch.equal(before[name], p.detach()) for name, p in after.items())
    _, loss_after, _ = model(tokens, targets)
    assert torch.isfinite(loss_after)


def test_live_profile_is_isolated_and_complete() -> None:
    assert {field.name for field in fields(VFE3Config)} == set(live_values())
    first = live_values()
    family = first["family"]
    first["family"] = "changed"
    assert live_values()["family"] == family
    assert config_values()["attention_mode"] == "projected_gradient"     # the legacy baseline is unchanged
    cfg = live_config()
    assert config_from_serialized(serialize_config(cfg), source="live profile") == cfg
    assert asdict(live_config(n_layers=1))["n_layers"] == 1 and live_config().n_layers == 2
