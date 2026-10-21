"""Cross-cutting checks on the live launcher route at tiny scale (``tests.profiles.live_config``).

The default test profile is the legacy projected_gradient route; these checks run the route
``train_vfe3.py`` trains. The profile takes every key of the launcher's ``config`` dictionary, shrinks the
shape and run fields to a CPU unit test, and pins the fields the launcher omits at test-owned values (their frozen
introduction values, ``vfe3.config_history``), so a route switch in the launcher reaches these checks without an
edit here. Every check here must hold for any valid launcher setting: where a check needs a toggle the launcher may
hold otherwise (the frame transport of the frame-exponential check), it sets it.
"""
import math
from dataclasses import asdict, fields
from typing import Any, Callable, Dict, Tuple

import pytest
import torch

from tests.profiles import _LIVE_TINY, config_values, launcher_values, live_config, live_kl_max, live_values
from vfe3.config import VFE3Config, config_from_serialized, serialize_config
from vfe3.config_history import FIELD_INTRODUCTION_VALUES
from vfe3.model.model import VFEModel
from vfe3.train import build_optimizer


def _model(seed: int = 0, **overrides: Any) -> VFEModel:
    torch.manual_seed(seed)
    return VFEModel(live_config(**overrides))


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
    # A field the launcher omits holds a test-owned value: its frozen introduction value (which a changed default
    # leaves alone); a field newer than that table holds the baseline value of tests.profiles.
    omitted = {name for name in values if name not in launcher and name not in resized}
    assert {name: (values[name], FIELD_INTRODUCTION_VALUES[name]) for name in omitted
            if name in FIELD_INTRODUCTION_VALUES and values[name] != FIELD_INTRODUCTION_VALUES[name]} == {}
    # kl_max keeps the launcher's nats per comparison coordinate: it scales by the head width d = K / H, d_tiny /
    # d_launcher, and a launcher None (8 K) is resolved at the launcher's K first (audit 2026-10-20 item 5).
    cfg = live_config()
    d_tiny = cfg.embed_dim / cfg.n_heads
    d_launcher = launcher["embed_dim"] / launcher["n_heads"]
    launcher_cap = 8.0 * launcher["embed_dim"] if launcher.get("kl_max") is None else launcher["kl_max"]
    assert values["kl_max"] == pytest.approx(launcher_cap * d_tiny / d_launcher, rel=1e-15)
    # the reference run (K = 160, five heads of 32, kl_max 128) and a K = 256 eight-head launcher both give 16
    assert live_kl_max({"kl_max": 128.0, "embed_dim": 160, "n_heads": 5}) == 16.0
    assert live_kl_max({"kl_max": 128.0, "embed_dim": 256, "n_heads": 8}) == 16.0
    assert live_kl_max({"kl_max": None, "embed_dim": 128, "n_heads": 4}) == 8.0 * 128 * d_tiny / 32


def _dead_router_maps(model: VFEModel) -> list:
    r"""The comparison maps whose gradient is absent or exactly zero after a backward."""
    grads = {name: p.grad for name, p in model.named_parameters() if "raw_score" in name and p.requires_grad}
    assert grads, "the live route has no comparison maps"
    return [name for name, grad in grads.items() if grad is None or not bool((grad != 0).any())]


def _with_a_live_router(build: Callable[..., VFEModel], **overrides: Any) -> Tuple[VFEModel, Callable[[], torch.Tensor]]:
    r"""``build(**overrides)`` and its loss closure on one fixed batch, backward taken, with a router that is not
    clamped dead.

    The tiny cap keeps the launcher's nats per comparison coordinate (:func:`tests.profiles.live_kl_max`). On a
    launcher route where that cap still binds on every pair of a layer (for instance the absolute reference on the
    capped pushforward of a full family), the router gradient is zero by the clamp's definition and a gradient
    check of the router would compare zero with zero (audit 2026-10-20 item 5); there the ceiling is lifted
    (``kl_max=inf``) so that the router's backward is still checked, on the same route otherwise.
    """
    for lifted in ({}, {"kl_max": math.inf}):
        model = build(**{**overrides, **lifted})
        tokens, targets = _batch(model.cfg)

        def loss(model: VFEModel = model, tokens: torch.Tensor = tokens, targets: torch.Tensor = targets
                 ) -> torch.Tensor:
            return model(tokens, targets)[1]

        loss().backward()
        if not _dead_router_maps(model):
            break
    _assert_router_gradient_nonzero(model)
    return model, loss


def _assert_router_gradient_nonzero(model: VFEModel) -> None:
    r"""Every comparison map (``raw_score_q``, ``raw_score_k``) receives a nonzero gradient after a backward: the
    tiny live cap does not clamp the router dead (audit 2026-10-20 item 5, where every live-route gradient check
    compared a router gradient of exactly zero)."""
    dead = _dead_router_maps(model)
    assert dead == [], f"router gradient is exactly zero on {dead}"


def test_live_forward_backward_is_finite() -> None:
    model, loss = _with_a_live_router(_model)
    assert torch.isfinite(loss())
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    # Every trainable parameter is reached by backward (audit 2026-10-16 F111): a dead table or a severed
    # detach leaves its gradient None.
    assert trainable and [name for name, p in trainable.items() if p.grad is None] == []
    grads = [p.grad for p in trainable.values()]
    assert all(bool(torch.isfinite(g).all()) for g in grads)
    assert any(bool((g != 0).any()) for g in grads)


def _float64_live_model(**overrides: Any) -> VFEModel:
    r"""The live route in float64 with untied block-norm gains and a random unigram table.

    The block-norm gains start tied at one; they are moved off the tie so that no maximum over the gains is
    differentiated at a kink. The unigram decode table is filled from random counts, so the unigram bias is
    nonzero on the checked route (audit 2026-10-17 item 26); an unset table adds zero.
    """
    torch.manual_seed(0)
    model = VFEModel(live_config(**overrides)).double()
    with torch.no_grad():
        generator = torch.Generator().manual_seed(5)
        for name, p in model.named_parameters():
            if name.endswith("norm.weight"):
                p.add_(0.01 * torch.randn(p.shape, dtype=p.dtype, generator=generator))
    bank = model.prior_bank
    if bank.decode_unigram_prior:
        counts = torch.randint(0, 50, (bank.vocab_size,), generator=torch.Generator().manual_seed(6))
        bank.set_unigram_log_prior(counts.double())
    return model


def _richardson_mismatches(
    loss:   Callable[[], torch.Tensor],
    params: Dict[str, torch.nn.Parameter],

    *,
    h:      float = 1e-4,
    seed:   int   = 2,
) -> Dict[str, Tuple[float, float, float]]:
    r"""Parameters whose analytic directional derivative misses a Richardson-extrapolated central difference.

    For one random direction :math:`d_p` per parameter, :math:`D(h) = (L(p + h d_p) - L(p - h d_p))/(2h)
    = \langle\nabla_p L, d_p\rangle + c h^2 + O(h^4)`, so :math:`R = (4D(h/2) - D(h))/3` removes the
    :math:`h^2` truncation that a single step cannot tell from a defect (audit 2026-10-17 item 37). The
    tolerance is :math:`|R - a| \le 10^{-6}\max(|R|, |a|) + \max(10^{-8}, 10^3\varepsilon|L|/h)`, the second
    term bounding the float64 cancellation in :math:`L(p + h d) - L(p - h d)`. Every ``p.grad`` must already
    hold :math:`\nabla_p L`.

    The remainder of :math:`R` is :math:`O(h^4)`, and its constant depends on the route (a launcher route with a large
    fourth derivative leaves a few :math:`10^{-6}` at :math:`h = 10^{-4}`). A parameter that misses at :math:`h` is
    therefore measured again at :math:`h/4`, where that remainder is 256 times smaller, and counts as mismatched
    unless it meets the tolerance there with a smaller error: a defect in the backward does not shrink with the step.
    """
    eps = torch.finfo(torch.float64).eps
    with torch.no_grad():
        base = abs(float(loss()))
    directions = torch.Generator().manual_seed(seed)
    mismatched: Dict[str, Tuple[float, float, float]] = {}
    for name, p in params.items():
        d = torch.randn(p.shape, dtype=p.dtype, generator=directions)
        original = p.detach().clone()

        def central(step: float) -> float:
            with torch.no_grad():
                p.copy_(original + step * d)
                plus = float(loss())
                p.copy_(original - step * d)
                minus = float(loss())
                p.copy_(original)
            return (plus - minus) / (2.0 * step)

        def miss(step: float) -> Tuple[float, float, float]:
            richardson = (4.0 * central(step / 2.0) - central(step)) / 3.0
            tolerance = 1e-6 * max(abs(richardson), abs(analytic)) + max(1e-8, 1e3 * eps * base / step)
            return richardson, abs(richardson - analytic), tolerance

        analytic = float((p.grad * d).sum())
        richardson, error, tolerance = miss(h)
        if error > tolerance:
            _, finer_error, finer_tolerance = miss(h / 4.0)
            if finer_error > finer_tolerance or finer_error >= error:
                mismatched[name] = (richardson, analytic, tolerance)
    return mismatched


@pytest.mark.parametrize("block_norm_covariance", ["passthrough", "pushforward"])
def test_live_backward_matches_float64_richardson_differences(block_norm_covariance: str) -> None:
    r"""Each trainable parameter's gradient on the live route matches a float64 Richardson-extrapolated central
    difference (audit 2026-10-16 F111, 2026-10-17 item 37), under both LayerNorm covariance maps.

    A wrong custom backward, a missing chain-rule term or a stray detach on the launcher route moves the analytic
    side away from the extrapolated difference, while the :math:`O(h^2)` truncation the pushforward's large third
    derivative produces at a single step does not.
    """
    model, loss = _with_a_live_router(_float64_live_model, block_norm_covariance=block_norm_covariance)
    assert model.cfg.block_norm_covariance == block_norm_covariance
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    assert trainable and [name for name, p in trainable.items() if p.grad is None] == []
    assert _richardson_mismatches(loss, trainable) == {}


def test_live_phi_embed_gradient_under_the_fixed_schedule_exponential(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""The CUDA-only fixed-schedule frame exponential, forced onto the CPU live route, gives a ``phi_embed``
    gradient that matches a Richardson difference with the frames ten times off the identity (audit 2026-10-17
    item 26).

    ``_frame_matrix_exp`` takes :class:`~vfe3.geometry.transport._FixedScheduleMatrixExp` only on CUDA; the
    patch applies the same rule on every device, so its custom backward joins the live route here.
    """
    from vfe3.geometry import transport

    calls = {"fixed": 0}

    def fixed_schedule(blocks: torch.Tensor, norm_bound: float) -> torch.Tensor:
        if not math.isfinite(norm_bound) or blocks.dtype not in (torch.float32, torch.float64):
            return torch.linalg.matrix_exp(blocks)
        calls["fixed"] += 1
        return transport._FixedScheduleMatrixExp.apply(blocks, math.sqrt(blocks.shape[-1]) * norm_bound)

    monkeypatch.setattr(transport, "_frame_matrix_exp", fixed_schedule)
    model = _float64_live_model(gauge_transport="on")        # the frame exponential runs only with transport on
    phi = model.prior_bank.phi_embed
    with torch.no_grad():
        phi.mul_(10.0)
    tokens, targets = _batch(model.cfg)

    def loss() -> torch.Tensor:
        return model(tokens, targets)[1]

    loss().backward()
    assert calls["fixed"] > 0, "the fixed-schedule exponential never ran on the live route"
    assert phi.grad is not None and bool((phi.grad != 0).any())
    assert _richardson_mismatches(loss, {"prior_bank.phi_embed": phi}) == {}


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
