r"""Bitwise performance changes of 2026-10-28 (docs/perf-investigation_2026-10-28.md, the small bitwise set).

* Rank 1: :func:`vfe3.model.frame_actions.identity_frame` builds its context without the float64 value validation
  of :class:`vfe3.contracts.CanonicalFrameContext` (an exact identity always passes it); real frames still validate.
* Rank 3: the fused cue/result memory allocates its fully overwritten planes uninitialized; seeding them with NaN
  changes nothing (interpreter subprocess).
* Rank 5: the table-Fisher frame capture runs inside :func:`vfe3.contracts.deferred_host_checks`.
* Rank 11: the fused memory leaves unused output gradients unmaterialized; a loss on the mean alone matches the
  torch route (interpreter subprocess).
* Rank 12: on the fused fp32 path the gradient-finiteness verdict comes from the transferred L2 set norms, with the
  exact check as the fallback when a norm is not finite; a NaN gradient still skips the step.
* Rank 17: :func:`vfe3.free_energy.attention_weights` at ``tau == 1.0`` is one subtraction, bitwise equal to
  ``softmax(-E / 1 + B)`` with its gradients.

Ranks 13 and 15 (no log weights in the backward rerun of the weights kernel; one forward normalization pass) and the
whole set together are checked bitwise against a pre-change snapshot of a tiny copy of the live route
(``docs/perf-2026-10-28/parity_snapshot.py``). Tiny CPU cases only.
"""
from __future__ import annotations

import math
from typing import Any, List

import pytest
import torch

import vfe3.contracts as contracts
import vfe3.train as T
from tests.test_cue_result_triton_20261025 import _NO_TRITON, _run_script, _subprocess_env
from tests.test_frame_diagonal_table_fisher_20261018 import TOKENS, _cfg, _model
from vfe3.contracts import CanonicalFrameContext
from vfe3.free_energy import attention_weights
from vfe3.model.frame_actions import identity_frame
from vfe3.train import build_optimizer, train_step


# ------------------------------------------------------------------------------------------------ rank 1

def test_identity_frame_skips_value_validation_and_keeps_its_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[int] = []
    original = CanonicalFrameContext.__post_init__
    monkeypatch.setattr(CanonicalFrameContext, "__post_init__", lambda self: (calls.append(1), original(self))[1])
    g = torch.Generator().manual_seed(0)
    U = torch.randn(2, 3, 2, 4, 4, generator=g) + 3.0 * torch.eye(4)
    frame = CanonicalFrameContext(forward=U, inverse=torch.linalg.inv(U), block_diagonal=True)
    assert calls == [1]
    ident = identity_frame(frame)
    assert calls == [1]                                             # no validation of the identity context
    assert isinstance(ident, CanonicalFrameContext) and ident.block_diagonal is True
    assert ident.forward.shape == U.shape and ident.forward.dtype == U.dtype
    assert torch.equal(ident.forward, torch.eye(4).expand(U.shape)) and ident.inverse is ident.forward


def test_real_frames_still_validate_their_values() -> None:
    U = torch.eye(4).expand(2, 4, 4).clone()
    U[0, 0, 0] = float("nan")
    with pytest.raises(Exception):
        CanonicalFrameContext(forward=U, inverse=U.clone(), block_diagonal=False)


# ------------------------------------------------------------------------------------------------ rank 17

@pytest.mark.parametrize("prior", ["same", "broadcast", "promote", "none"])
def test_attention_weights_at_unit_tau_is_bitwise_the_reference(prior: str) -> None:
    g = torch.Generator().manual_seed(3)
    energy = torch.randn(2, 3, 5, 5, generator=g, dtype=torch.float32)
    log_prior = {"same": torch.randn(2, 3, 5, 5, generator=g), "broadcast": torch.randn(3, 5, 5, generator=g),
                 "promote": torch.randn(2, 3, 5, 5, generator=g, dtype=torch.float64), "none": None}[prior]
    e1 = energy.clone().requires_grad_(True)
    e2 = energy.clone().requires_grad_(True)
    p1 = None if log_prior is None else log_prior.clone().requires_grad_(True)
    p2 = None if log_prior is None else log_prior.clone().requires_grad_(True)
    ref_logits = -e1 / 1.0 if p1 is None else -e1 / 1.0 + p1
    ref = torch.softmax(ref_logits, dim=-1)
    got = attention_weights(e2, tau=1.0, log_prior=p2)
    assert got.dtype == ref.dtype and torch.equal(got, ref)
    cot = torch.randn(ref.shape, generator=g, dtype=ref.dtype)
    (ref * cot).sum().backward()
    (got * cot).sum().backward()
    assert torch.equal(e1.grad, e2.grad)
    if p1 is not None:
        assert torch.equal(p1.grad, p2.grad)


# ------------------------------------------------------------------------------------------------ rank 5

def test_table_fisher_frame_capture_runs_in_a_deferred_host_check_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg()
    model = _model(cfg)
    optimizer = build_optimizer(model, cfg)
    provider = optimizer.table_statistics_provider
    assert provider.reads_frames
    scoped: List[bool] = []
    original = provider.capture_frames
    monkeypatch.setattr(provider, "capture_frames",
                        lambda: (scoped.append(contracts._DEFERRED_CHECKS is not None), original())[1])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    train_step(model, optimizer, scheduler, TOKENS, TOKENS.roll(-1, 1), grad_clip=1.0)
    assert scoped == [True]


# ------------------------------------------------------------------------------------------------ rank 12

def _step(cfg: Any, poison: Any, monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype = torch.float32) -> tuple:
    model = _model(cfg).to(dtype)
    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    exact_calls: List[int] = []
    original = T._gradients_finite
    monkeypatch.setattr(T, "_gradients_finite", lambda grads: (exact_calls.append(1), original(grads))[1])
    if poison:
        value = float("nan") if poison is True else float(poison)
        param = next(p for p in model.parameters() if p.requires_grad)
        param.register_hook(lambda grad: torch.full_like(grad, value))
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    status: dict = {}
    train_step(model, optimizer, scheduler, TOKENS, TOKENS.roll(-1, 1), grad_clip=1.0, status_out=status)
    moved = any(not torch.equal(before[n], p.detach()) for n, p in model.named_parameters())
    return exact_calls, moved, status


def test_finite_gradients_take_the_verdict_from_the_set_norms(monkeypatch: pytest.MonkeyPatch) -> None:
    calls, moved, _ = _step(_cfg(), poison=False, monkeypatch=monkeypatch)
    assert calls == [] and moved


def test_a_nan_gradient_falls_back_to_the_exact_check_and_skips_the_step(monkeypatch: pytest.MonkeyPatch) -> None:
    calls, moved, _ = _step(_cfg(), poison=True, monkeypatch=monkeypatch)
    assert calls == [1] and not moved


# ------------------------------------------------------------------------------------------------ ranks 3 and 11

_FUSED_SCRIPT = """
import json, math, sys, warnings
try:
    import triton  # noqa: F401
except Exception as exc:
    print("triton import failed:", exc)
    sys.exit(77)
import torch
torch.set_num_threads(1)
warnings.simplefilter("ignore")
from vfe3.kernels import _cue_result_triton as tk
from vfe3.kernels import cue_result as fused_kernels
from vfe3.model.cue_result_memory import cue_result_diagonal_posterior
assert tk.INTERPRETED
fused_kernels.fused_route_active = lambda xi: True

_empty = torch.empty
def nan_empty(*args, **kwargs):
    out = _empty(*args, **kwargs)
    return out.fill_(float("nan")) if out.is_floating_point() else out

g = torch.Generator().manual_seed(5)
R, M, K, rank = 5, 9, 37, 2
d = torch.float64
ops = {"xi": torch.randn(R, K, generator=g, dtype=d), "c": 0.3 + torch.rand(R, K, generator=g, dtype=d),
       "nu": torch.randn(M, K, generator=g, dtype=d), "omega": torch.randn(M, K, generator=g, dtype=d),
       "log_lambda": 0.5 * torch.randn(M, K, generator=g, dtype=d),
       "log_result_noise": 0.5 * torch.randn(M, K, generator=g, dtype=d) - 1.0,
       "log_pi": torch.randn(M, generator=g, dtype=d),
       "regression_left": 0.7 * torch.randn(M, K, rank, generator=g, dtype=d),
       "regression_right": 0.7 * torch.randn(M, K, rank, generator=g, dtype=d)}

def run(fused, mean_only, nan_fill):
    leaves = {k: v.clone().requires_grad_(True) for k, v in ops.items()}
    torch.empty = nan_empty if nan_fill else _empty
    try:
        out = cue_result_diagonal_posterior(*(leaves[k] for k in ("xi", "c", "nu", "omega", "log_lambda",
                                              "log_result_noise", "log_pi")),
                                            regression_left=leaves["regression_left"],
                                            regression_right=leaves["regression_right"], fused=fused)
        loss = out.mean.square().sum() + (0 if mean_only else out.covariance.sum() + out.log_weights.exp().sum()
                                          + out.log_evidence.sum())
        grads = torch.autograd.grad(loss, list(leaves.values()), allow_unused=True)
        grads = [torch.zeros_like(v) if g is None else g for g, v in zip(grads, leaves.values())]
    finally:
        torch.empty = _empty
    return [out.mean, out.covariance, *grads]

def rel(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-300))

report = {}
for mean_only in (False, True):
    ref = run(False, mean_only, False)
    clean = run(True, mean_only, False)
    seeded = run(True, mean_only, True)
    key = "mean_only" if mean_only else "all_outputs"
    report[key] = {"vs_torch": max(rel(a, b) for a, b in zip(clean, ref)),
                   "nan_seeded_bitwise": all(torch.equal(a, b) for a, b in zip(seeded, clean)),
                   "finite": all(bool(torch.isfinite(t).all()) for t in seeded)}
print(json.dumps(report))
"""


def test_fused_memory_planes_need_no_initialization_and_unused_grads_stay_unmaterialized() -> None:
    report = _run_script(_FUSED_SCRIPT, _subprocess_env(TRITON_INTERPRET="1"))
    if report == _NO_TRITON:
        pytest.skip("triton unavailable")
    for key in ("all_outputs", "mean_only"):
        assert report[key]["nan_seeded_bitwise"] and report[key]["finite"], report
        assert report[key]["vs_torch"] < 1e-11, report


def test_a_float64_gradient_beyond_the_float32_range_takes_the_exact_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    # The exact check reads each infinity norm in float32, so 1e39 counts as nonfinite there; the float64 set norm is
    # finite but above the float32 maximum, so the exact check decides and the step is skipped as before.
    calls, moved, _ = _step(_cfg(), poison=1e39, monkeypatch=monkeypatch, dtype=torch.float64)
    assert calls == [1] and not moved


# ------------------------------------------------------------------------------------------------ rank 4

@pytest.mark.parametrize("scale,trace", [(1.0, 0.0), (30.0, 0.0), (5.0, 40.0), (1e-3, 0.0)])
def test_frame_screen_doubled_blocks_are_twice_the_traceless_blocks_bit_for_bit(scale: float, trace: float) -> None:
    from vfe3.gauge_optim import _doubled_traceless_symmetric, _traceless_symmetric_blocks
    g = torch.Generator().manual_seed(11)
    H, d = 2, 8
    phi = scale * torch.randn(40, H * d * d, generator=g)
    phi.view(40, H, d, d).diagonal(dim1=-2, dim2=-1).add_(trace)            # a large trace stresses cancellation
    S0 = _traceless_symmetric_blocks(phi, None, H, d, torch.float32)
    T = _doubled_traceless_symmetric(phi, None, H, d, torch.float32)
    assert torch.equal(2.0 * S0, T)
    assert torch.equal(4.0 * S0.square().sum(dim=(-2, -1)), T.square().sum(dim=(-2, -1)))
