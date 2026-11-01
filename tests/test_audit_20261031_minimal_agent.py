r"""Remediation package ``minimal-agent`` of the October 31, 2026 audit (tiny CPU cases only).

A014/A015 (gap-checkpoint-recompute-semantics-1/2): the exact and label-spread tiles evaluate through a pinned
recompute, so their recompute and backward GEMMs stay IEEE fp32 under ``fp32_matmul_precision="high"``; the routed
spread tiles carry the pin themselves. The CPU has no TF32, so every float32 GEMM entry point is routed through an
emulation that rounds its operands to TF32 whenever the policy at the call is not ``"highest"`` (the audit lane's
symmetric emulation). A026 (minimal-agent-receiver-1): an exactly singular float32 factor is a failure flag, not a
``LinAlgError``. A090 (minimal-agent-receiver-2): a masked pair or an unread utterance covariance raises no flag, so a
later token cannot move an earlier receiver. A078 (gap-certified-factorization-routes-3): the certified mean field
reads its shared factors, and the run record names no KL refactor. A177 (block-mlp-norm-6): the head-block ceiling
makes one call per head width. A192 (minimal-agent-receiver-4): a partly masked tile factors only admitted pairs,
bitwise the dense tile. A194 (minimal-agent-receiver-6): the tile plan is cached per prior and view, and the
frame-diagonal body skips its identity products bitwise.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import pytest
import torch

import vfe3.model.covariance_ceiling as ceiling_module
import vfe3.model.minimal_agent as ma
import vfe3.model.relational_operators as ro
from tests.profiles import make_relational_operators, tiny_config
from vfe3.contracts import CanonicalFrameContext
from vfe3.model.frame_actions import identity_frame
from vfe3.relational_scoring import (
    reset_source_recovery_elements, set_projected_condition_recovery, source_recovery_elements,
)
from vfe3.runtime import ieee_fp32_matmul

torch.set_num_threads(1)
F64 = torch.float64


@pytest.fixture(autouse=True)
def _no_condition_recovery() -> Iterator[None]:
    r"""Every case below runs with the certificate off unless it switches it on itself (process-global state)."""
    previous = set_projected_condition_recovery(False)
    yield
    set_projected_condition_recovery(previous)


# ----------------------------------------------------------------------------------------------
# Symmetric TF32 emulation of the CPU GEMM entry points (A014, A015)
# ----------------------------------------------------------------------------------------------

def _tf32_operand(x: Any) -> Any:
    r"""A float32 tensor rounded to TF32 (10 stored mantissa bits, round to nearest even), detached; others as is."""
    if not torch.is_tensor(x) or x.dtype != torch.float32:
        return x
    bits = x.detach().contiguous().view(torch.int32)
    lsb = (bits >> 13) & 1
    return ((bits + 0x0FFF + lsb) & ~0x1FFF).view(torch.float32)


def _tf32_active() -> bool:
    return torch.get_float32_matmul_precision() != "highest"


class _EmulatedGemm(torch.autograd.Function):
    r"""A float32 GEMM ``fn(*ops)`` whose operands round to TF32 unless the policy at the call is ``"highest"``; its
    backward reads the policy again when it runs and treats the gradient GEMMs alike."""

    @staticmethod
    def forward(ctx, fn: Callable[..., torch.Tensor], *ops: torch.Tensor) -> torch.Tensor:
        ctx.fn = fn
        ctx.save_for_backward(*ops)
        return fn(*([_tf32_operand(o) for o in ops] if _tf32_active() else ops))

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        tf32 = _tf32_active()
        leaves = [(_tf32_operand(o) if tf32 else o).detach().requires_grad_(need)
                  for o, need in zip(ctx.saved_tensors, ctx.needs_input_grad[1:])]
        with torch.enable_grad():
            out = ctx.fn(*leaves)
            wanted = [x for x in leaves if x.requires_grad]
            grads = iter(torch.autograd.grad(out, wanted, _tf32_operand(grad) if tf32 else grad, allow_unused=True))
        return (None, *(next(grads) if x.requires_grad else None for x in leaves))


@pytest.fixture()
def tf32(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    r"""Process policy ``"high"`` with every float32 einsum, matmul, bmm and ``@`` emulated at TF32 under it."""
    originals = {"einsum": torch.einsum, "matmul": torch.matmul, "bmm": torch.bmm,
                 "__matmul__": torch.Tensor.__matmul__}

    def route(fn: Callable[..., torch.Tensor], ops: Sequence[Any]) -> torch.Tensor:
        if (torch.is_grad_enabled() and any(torch.is_tensor(o) and o.requires_grad for o in ops)
                and any(torch.is_tensor(o) and o.dtype == torch.float32 for o in ops)):
            return _EmulatedGemm.apply(fn, *ops)
        return fn(*([_tf32_operand(o) for o in ops] if _tf32_active() else ops))

    def einsum(equation: str, *ops: Any) -> torch.Tensor:
        if len(ops) == 1 and isinstance(ops[0], (list, tuple)):
            ops = tuple(ops[0])
        return route(lambda *xs: originals["einsum"](equation, *xs), ops)

    monkeypatch.setattr(torch, "einsum", einsum)
    monkeypatch.setattr(torch, "matmul", lambda a, b: route(originals["matmul"], (a, b)))
    monkeypatch.setattr(torch, "bmm", lambda a, b: route(originals["bmm"], (a, b)))
    monkeypatch.setattr(torch.Tensor, "__matmul__", lambda a, b: route(originals["__matmul__"], (a, b)))
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


def _spd64(n: int, batch: Tuple[int, ...], gen: torch.Generator, scale: float) -> torch.Tensor:
    m = torch.randn(*batch, n, n, generator=gen, dtype=F64)
    return scale * (m @ m.transpose(-1, -2) / n + 0.5 * torch.eye(n, dtype=F64))


def _grads(loss: Callable[..., torch.Tensor], inputs: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    r"""Forward under the IEEE pin (as the readouts pin it), backward outside it at the process policy."""
    leaves = [t.detach().clone().requires_grad_(True) for t in inputs]
    with ieee_fp32_matmul():
        value = loss(*leaves)
    value.backward()
    return [t.grad for t in leaves]


def _max_rel(got: Sequence[torch.Tensor], want: Sequence[torch.Tensor]) -> float:
    return max(float((g.double() - w).norm() / w.norm()) for g, w in zip(got, want))


_U_DIR = torch.tensor([1.0, 0.6], dtype=F64)
_W_GEN = torch.Generator().manual_seed(1)
_W_U = torch.randn(1, 6, 2, generator=_W_GEN, dtype=F64)
_W_L = torch.randn(1, 6, 2, 2, generator=_W_GEN, dtype=F64)
_W_R = torch.randn(1, 6, 2, 2, generator=_W_GEN, dtype=F64)
_W_S = torch.randn(1, 12, 2, 2, generator=_W_GEN, dtype=F64)


def _exact_inputs(offset: float) -> List[torch.Tensor]:
    r"""Six receivers and twelve sources of rank 2, the sources offset by ``offset`` source spreads."""
    gen = torch.Generator().manual_seed(2)
    a = 0.3 * torch.randn(1, 6, 2, generator=gen, dtype=F64)
    A = _spd64(2, (1, 6), gen, 0.5)
    b = offset * _U_DIR + 0.3 * torch.randn(1, 12, 2, generator=gen, dtype=F64)
    B = _spd64(2, (1, 12), gen, 0.5)
    return [a, A, b, B]


def _exact_loss(a: torch.Tensor, A: torch.Tensor, b: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    post = ma.exact_record_posterior(a, A, b, B, None, check=False)
    d = a.dtype
    return (post.u * _W_U.to(d)).sum() + (post.Lambda_bar * _W_L.to(d)).sum() + (post.R * _W_R.to(d)).sum()


@pytest.mark.parametrize("tiled", [False, True])
def test_exact_tile_gradients_stay_ieee_under_the_tf32_policy(tf32: None, tiled: bool,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    r"""At an offset of 30 spreads the shipped tile's gradient was 26 percent (untiled) or 13 percent (checkpointed)
    off float64 under emulated TF32; the pinned recompute keeps it at float32 accuracy (audit A014, A015)."""
    if tiled:
        monkeypatch.setattr(ma, "PAIR_TILE_BYTES", 1152)              # three float32 source tiles of four
    inputs = _exact_inputs(30.0)
    reference = _grads(_exact_loss, inputs)
    assert _max_rel(_grads(_exact_loss, [t.float() for t in inputs]), reference) < 1e-4


def _spread_inputs(offset: float) -> List[torch.Tensor]:
    r"""Twelve receivers and sources shifted together by ``offset`` source spreads."""
    gen = torch.Generator().manual_seed(3)
    logits = torch.randn(1, 12, 12, generator=gen, dtype=F64)
    a = offset * _U_DIR + 0.3 * torch.randn(1, 12, 2, generator=gen, dtype=F64)
    b = 0.3 * torch.randn(1, 12, 2, generator=gen, dtype=F64) + offset * _U_DIR
    B = _spd64(2, (1, 12), gen, 0.5)
    return [logits, a, b, B]


def _spread_loss(logits: torch.Tensor, a: torch.Tensor, b: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    R, _ = ma.label_spread(torch.softmax(logits, dim=-1), a, b, B)
    return (R * _W_S.to(a.dtype)).sum()


@pytest.mark.parametrize("tiled", [False, True])
def test_label_spread_gradients_stay_ieee_under_the_tf32_policy(tf32: None, tiled: bool,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    if tiled:
        monkeypatch.setattr(ma, "SPREAD_TILE_BYTES", 200)             # six float32 tiles of two receiver rows
    inputs = _spread_inputs(30.0)
    reference = _grads(_spread_loss, inputs)
    assert _max_rel(_grads(_spread_loss, [t.float() for t in inputs]), reference) < 1e-4


def _routed_inputs() -> List[torch.Tensor]:
    gen = torch.Generator().manual_seed(4)
    logits = torch.randn(1, 2, 6, 6, generator=gen)
    values = 30.0 + 0.3 * torch.randn(1, 6, 2, 2, generator=gen)
    lift = torch.randn(1, 6, 2, 4, 2, generator=gen)
    return [logits, values, lift]


def _squares_loss(logits: torch.Tensor, values: torch.Tensor, lift: torch.Tensor) -> torch.Tensor:
    beta = torch.softmax(logits, dim=-1)
    mean = torch.einsum("...hij,...jhs->...ihs", beta, values)
    return ro._spread_squares(beta, values, mean, lift, "diag").square().sum()


def _centered_loss(logits: torch.Tensor, values: torch.Tensor, lift: torch.Tensor) -> torch.Tensor:
    beta = torch.softmax(logits, dim=-1)
    mean = torch.einsum("...hij,...jhs->...ihs", beta, values)
    return ro._centered_spread(beta, values, mean).square().sum()


@pytest.mark.parametrize("loss", [_squares_loss, _centered_loss])
def test_routed_spread_recompute_reproduces_its_forward_under_the_tf32_policy(
        tf32: None, loss: Callable[..., torch.Tensor], monkeypatch: pytest.MonkeyPatch) -> None:
    r"""Real checkpoints against pass-through ones: bitwise equal gradients once the tiles carry the pin (A014)."""
    previous = ro.set_spread_tile_bytes(1)                            # one checkpointed query row per tile
    try:
        inputs = _routed_inputs()
        real = _grads(loss, inputs)
        monkeypatch.setattr(ro, "checkpoint", lambda fn, *args, **kwargs: fn(*args))
        passed = _grads(loss, inputs)
    finally:
        ro.set_spread_tile_bytes(previous)
    assert all((x is None and y is None) or torch.equal(x, y) for x, y in zip(real, passed))


def test_checkpointed_tiles_carry_the_ieee_pin() -> None:
    r"""A function decorated by the ``ieee_fp32_matmul`` context manager carries ``__wrapped__``."""
    for tile in (ma._exact_tile, ma._label_spread_tile, ro._spread_squares_tile, ro._centered_spread_tile):
        assert hasattr(tile, "__wrapped__"), tile.__name__


# ----------------------------------------------------------------------------------------------
# A026: an exactly singular float32 factor is a flag
# ----------------------------------------------------------------------------------------------

# [[4e8, 2e8], [2e8, 1e8]] is exact in float32 and rank one: its second Cholesky pivot is exactly 0.
_RANK_ONE = torch.tensor([[4.0e8, 2.0e8], [2.0e8, 1.0e8]])


def test_exact_record_posterior_flags_an_exactly_singular_record_covariance() -> None:
    a, b = torch.zeros(1, 2), torch.ones(1, 2)
    A, B = _RANK_ONE.unsqueeze(0), torch.zeros(1, 2, 2)              # W = A + B rank one
    post = ma.exact_record_posterior(a, A, b, B, check=False)
    assert bool(post.failed) and post.row_failed.tolist() == [True]
    for value in (post.weights, post.cost, post.u, post.Lambda_bar, post.R):
        assert bool(torch.isfinite(value).all())
    with pytest.raises(ValueError, match="record covariance"):
        ma.exact_record_posterior(a, A, b, B)


@pytest.mark.parametrize("singular", ["kalman", "precision"])
def test_information_step_flags_an_exactly_singular_factor(singular: str) -> None:
    r"""E = I + L^T A L rounds to the rank-one A in float32 (``kalman``), or Lambda itself is rank one."""
    Lam, A = ((torch.eye(2), _RANK_ONE) if singular == "kalman" else (_RANK_ONE, torch.eye(2)))
    u, M, failed = ma.information_step(torch.zeros(1, 2), A.unsqueeze(0), Lam.unsqueeze(0), torch.ones(1, 2))
    assert failed.tolist() == [True]
    assert bool(torch.isfinite(u).all()) and bool(torch.isfinite(M).all())


# The audit's N = 3, H = 1, d = 2 float32 fixture: token 2 alone takes an ill-conditioned covariance.
_S = torch.tensor([[[1.0, 0.2], [-0.1, 1.0]]])
_T = torch.tensor([[[0.8, -0.3], [0.1, 1.1]]])
_MU = torch.tensor([[0.3, -0.7], [1.1, 0.2], [-0.5, 0.4]])
_BASE_C = torch.tensor([[0.7, 1.3], [1.2, 0.8], [0.9, 1.1]])


def _rotated(s: float, theta: float) -> torch.Tensor:
    c, sn = math.cos(theta), math.sin(theta)
    Q = torch.tensor([[c, -sn], [sn, c]], dtype=F64)
    return (Q @ torch.diag(torch.tensor([s, 1.0 / s], dtype=F64)) @ Q.T).float()


def _covariances(s: Optional[float] = None, theta: float = 0.0) -> torch.Tensor:   # (3, 2, 2)
    C = torch.diag_embed(_BASE_C).clone()
    if s is not None:
        C[2] = _rotated(s, theta)
    return C


def _causal(noself: bool = False) -> torch.Tensor:                   # (1, 3, 3)
    admit = torch.ones(3, 3, dtype=torch.bool).tril(-1 if noself else 0)
    if noself:
        admit[0, 0] = True                                            # row 0 keeps only its self entry
    return torch.zeros(1, 3, 3).masked_fill(~admit, -math.inf)


def _operator(inference: str, log_noise: float) -> ro.RelationalOperators:
    ops = make_relational_operators([2], 0, attention_residual=False, predictive_readout="minimal_agent",
                                    minimal_agent_inference=inference, predictive_covariance="moment_matched",
                                    predictive_covariance_floor=1e-4)
    with torch.no_grad():
        ops.raw_score_q.copy_(_S)
        ops.T.copy_(_T)
        ops.channel_log_noise.fill_(log_noise)
    return ops


def _update(ops: ro.RelationalOperators, C: torch.Tensor, lp: torch.Tensor,
            capture: Optional[dict] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""The local update with the identity frame, so the local beliefs are the reference moments."""
    eye = torch.eye(2).expand(1, 3, 2, 2)
    frame = CanonicalFrameContext(eye, eye, block_diagonal=False)
    with torch.no_grad():
        return ops.minimal_agent_update(_MU.unsqueeze(0), C.unsqueeze(0), frame, lp, return_covariance=True,
                                        floor_anchor=eye.clone(), capture=capture)


def _readout(inference: str, log_noise: float, C: torch.Tensor, lp: torch.Tensor) -> ma.MinimalAgentResult:
    return ma.minimal_agent_readout(_MU, C, _S, _T, torch.tensor([log_noise]), lp, inference, check=False)


@pytest.mark.parametrize("inference, log_noise, exponent, k", [("exact", -1.0, 80, 12), ("mean_field", -1.0, 78, 4)])
@pytest.mark.parametrize("diagnostics", [False, True])
def test_the_operator_completes_on_an_exactly_singular_float32_suffix(inference: str, log_noise: float,
                                                                      exponent: int, k: int,
                                                                      diagnostics: bool) -> None:
    r"""The first raising suffix of the audit's scan, per mode: the shipped operator raised ``LinAlgError`` from
    ``cholesky_inverse``; it now flags the float32 call (the mean field's ``E`` of receiver 2) or, on the exact path,
    no longer factors the masked pair W_02 whose pivot was zero, and completes with finite outputs."""
    C = _covariances(10.0 ** (exponent / 10), k * math.pi / 24)
    result = _readout(inference, log_noise, C, _causal())
    if inference == "mean_field":
        assert bool(result.failed) and result.flagged_receivers.tolist() == [False, False, True]
    mean, cov = _update(_operator(inference, log_noise), C, _causal(), capture={} if diagnostics else None)
    assert bool(torch.isfinite(mean).all()) and bool(torch.isfinite(cov).all())
    if inference == "mean_field" and diagnostics:
        # The gain-fraction diagnostic factors I + L^T A L at the input moments too; a failed factor reads NaN.
        traced = ma.minimal_agent_readout(_MU, C, _S, _T, torch.tensor([log_noise]), _causal(), inference,
                                          check=False, diagnostics=True)
        assert bool(torch.isfinite(traced.gain_fraction[:2]).all())


# ----------------------------------------------------------------------------------------------
# A090: masked pairs and unread sources raise no flag
# ----------------------------------------------------------------------------------------------

def test_a_failed_masked_pair_does_not_flag_its_receiver() -> None:
    r"""Receiver 0 admits only source 0; its masked pair W_01 = I + B_1 rounds to the rank-one B_1 in float32."""
    lp = torch.tensor([[0.0, -math.inf], [0.0, 0.0]])
    a, b = torch.zeros(2, 2), torch.ones(2, 2)
    A = torch.eye(2).expand(2, 2, 2).clone()
    B = torch.stack([torch.eye(2), _RANK_ONE])
    post = ma.exact_record_posterior(a, A, b, B, lp, check=False)
    assert post.row_failed.tolist() == [False, True]                  # receiver 1 reads B_1 through W_11
    assert float(post.cost[0, 1]) == 0.0 and float(post.weights[0, 1]) == 0.0


@pytest.mark.parametrize("exponent, k", [(85, 11), (90, 13)])
def test_exact_suffix_failure_leaves_the_prefix_unflagged_and_bitwise(exponent: int, k: int) -> None:
    r"""The audit's prefix fixture (N = 3, causal): the masked pairs W_02 and W_12 failed and flagged receivers 0 and
    1 although token 2 itself was healthy; now nothing is flagged, and the prefix outputs equal the benign suffix's
    bitwise."""
    C = _covariances(10.0 ** (exponent / 10), k * math.pi / 24)
    assert _readout("exact", -1.0, C, _causal()).flagged_receivers.tolist() == [False, False, False]
    ops = _operator("exact", -1.0)
    mean_a, cov_a = _update(ops, _covariances(), _causal())
    mean_b, cov_b = _update(ops, C, _causal())
    assert torch.equal(mean_a[:, :2], mean_b[:, :2]) and torch.equal(cov_a[:, :2], cov_b[:, :2])


def test_an_unread_utterance_failure_raises_no_flag() -> None:
    r"""Mean field under a self-masking prior: token 2 is admitted by no receiver, so its failed B_2 changes no output;
    it used to raise the flag with an empty receiver set, and the operator then recomputed every receiver."""
    C = _covariances(10.0 ** (64 / 10), 3 * math.pi / 24)
    for spread_corrected in (False, True):                             # the label spread read the same B_2 flag
        result = ma.minimal_agent_readout(_MU, C, _S, _T, torch.tensor([-4.0]), _causal(noself=True), "mean_field",
                                          check=False, spread_corrected=spread_corrected)
        assert not bool(result.failed) and not bool(result.flagged_receivers.any())
    ops = _operator("mean_field", -4.0)
    mean_a, cov_a = _update(ops, _covariances(), _causal(noself=True))
    mean_b, cov_b = _update(ops, C, _causal(noself=True))
    assert torch.equal(mean_a[:, :2], mean_b[:, :2]) and torch.equal(cov_a[:, :2], cov_b[:, :2])


# ----------------------------------------------------------------------------------------------
# A078: the certified mean field reads its shared factors; the record names no KL refactor
# ----------------------------------------------------------------------------------------------

def _mean_field_case() -> Tuple[torch.Tensor, ...]:
    gen = torch.Generator().manual_seed(5)
    N, K, H, d = 4, 4, 2, 2
    xi = torch.randn(N, K, generator=gen)
    X = torch.randn(N, K, K, generator=gen)
    C = X @ X.transpose(-1, -2) / K + 0.5 * torch.eye(K)
    S, T = torch.randn(H, d, K, generator=gen), torch.randn(H, d, K, generator=gen)
    lp = torch.zeros(N, N).masked_fill(torch.ones(N, N, dtype=torch.bool).triu(1), -math.inf)
    return xi, 0.5 * (C + C.transpose(-1, -2)), S, T, torch.full((H,), -0.5), lp


def test_certified_mean_field_reads_the_shared_factors(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[int] = []
    real = ma.projected_gaussian_kl
    monkeypatch.setattr(ma, "projected_gaussian_kl", lambda *a, **k: calls.append(1) or real(*a, **k))
    case = _mean_field_case()
    plain = ma.minimal_agent_readout(*case, "mean_field", check=False, cavi_steps=2)
    set_projected_condition_recovery(True)
    certified = ma.minimal_agent_readout(*case, "mean_field", check=False, cavi_steps=2)
    assert calls == []                                                 # no refactor, no scorer host read
    for name in ("xi", "gains", "M", "weights", "cost"):
        assert torch.equal(getattr(certified, name), getattr(plain, name)), name


def test_minimal_agent_factorization_record_names_no_kl_refactor() -> None:
    from vfe3.run_artifacts import _relational_artifact_fields
    for inference in ("exact", "mean_field"):
        cfg = tiny_config(attention_mode="projected_predictive", predictive_readout="minimal_agent",
                          minimal_agent_inference=inference)
        record = _relational_artifact_fields(cfg)["relational_operators"]
        assert record["factorization_fallback"] == "fp32_failure_or_certificate_layer_fp64_recompute"


def test_certified_mean_field_escalation_counts_every_entry_once() -> None:
    r"""A certificate-triggered float64 recompute counts the call's 2 B N H query and key entries, nothing else."""
    ops = _operator("mean_field", -0.4)
    C = torch.diag_embed(torch.tensor([[0.7, 1.3], [1.2, 0.8], [1e4, 1.0]]))
    set_projected_condition_recovery(True)
    reset_source_recovery_elements()
    _, _ = _update(ops, C, _causal())
    assert source_recovery_elements() in (0, 2 * 3 * 1)


# ----------------------------------------------------------------------------------------------
# A177: one ceiling call per head width
# ----------------------------------------------------------------------------------------------

def _block_spd(gen: torch.Generator, head_dims: Sequence[int], scales: Sequence[float]) -> torch.Tensor:
    K = sum(head_dims)
    out = torch.zeros(2, 3, K, K)
    start = 0
    for d, scale in zip(head_dims, scales):
        X = torch.randn(2, 3, d, d, generator=gen)
        out[..., start:start + d, start:start + d] = scale * (X @ X.transpose(-1, -2) / d + 0.3 * torch.eye(d))
        start += d
    return 0.5 * (out + out.transpose(-1, -2))


_DENSE_CEILING = ceiling_module._dense_ceiling


def _per_head_ceiling(sigma: torch.Tensor, anchor: torch.Tensor, c: float, head_dims: Sequence[int]) -> torch.Tensor:
    r"""The per-head loop the stacked call replaces."""
    anchor = torch.broadcast_to(anchor.detach().to(sigma.dtype), sigma.shape)
    out = sigma.clone()
    start = 0
    for d in head_dims:
        block = slice(start, start + d)
        out[..., block, block] = _DENSE_CEILING(sigma[..., block, block], anchor[..., block, block], c)
        start += d
    return out


@pytest.mark.parametrize("head_dims", [(2, 2, 2), (2, 3, 2)])
def test_head_block_ceiling_makes_one_call_per_head_width(head_dims: Tuple[int, ...],
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    gen = torch.Generator().manual_seed(6)
    sigma0 = _block_spd(gen, head_dims, (6.0, 0.5, 6.0))               # heads 0 and 2 exceed 2 Sigma_p somewhere
    anchor = _block_spd(gen, head_dims, (1.0, 1.0, 1.0))
    weight = torch.randn(sigma0.shape, generator=gen)
    results = []
    for stacked in (False, True):
        calls: List[int] = []
        monkeypatch.setattr(ceiling_module, "_dense_ceiling",
                            lambda *a: calls.append(1) or _DENSE_CEILING(*a))
        ceiling_module.reset_covariance_ceiling_elements()
        sigma = sigma0.clone().requires_grad_(True)
        out = (ceiling_module.covariance_ceiling(sigma, anchor, 2.0, diagonal=False, head_dims=head_dims)
               if stacked else _per_head_ceiling(sigma, anchor, 2.0, head_dims))
        (out * weight).sum().backward()
        results.append((out.detach(), sigma.grad, ceiling_module.covariance_ceiling_elements(), len(calls)))
    (out_a, grad_a, count_a, _), (out_b, grad_b, count_b, calls_b) = results
    assert calls_b == len(set(head_dims))
    assert count_a == count_b > 0
    assert torch.equal(out_a, out_b) and torch.equal(grad_a, grad_b)


# ----------------------------------------------------------------------------------------------
# A192 and A194: the cached tile plan, packed partly masked tiles, the identity frame
# ----------------------------------------------------------------------------------------------

# Float64 pair-tile bytes giving a budget of 12 pairs at batch 2 and rank 2 (q_step 6, k_step 2 for six receivers).
_TWO_SOURCE_TILES = 12 * 8 * 2 * 3 * 4


def _pair_case(gen_seed: int = 8, N: int = 6, batch: int = 2) -> List[torch.Tensor]:
    gen = torch.Generator().manual_seed(gen_seed)
    a, b = torch.randn(batch, N, 2, generator=gen, dtype=F64), torch.randn(batch, N, 2, generator=gen, dtype=F64)
    A, B = _spd64(2, (batch, N), gen, 0.7), _spd64(2, (batch, N), gen, 0.7)
    lp = (0.3 * torch.randn(N, N, generator=gen, dtype=F64)).masked_fill(
        torch.ones(N, N, dtype=torch.bool).triu(1), -math.inf)
    return [a, A, b, B, lp]


def _posterior_and_grads(inputs: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    leaves = [t.detach().clone().requires_grad_(True) for t in inputs]
    post = ma.exact_record_posterior(*leaves[:4], leaves[4], check=False)
    gen = torch.Generator().manual_seed(9)
    loss = sum((value * torch.randn(value.shape, generator=gen, dtype=F64)).sum()
               for value in (post.u, post.Lambda_bar, post.R, post.weights))
    loss.backward()
    lp_grad = leaves[4].grad.masked_fill(~torch.isfinite(inputs[4]), 0.0)
    return [post.weights.detach(), post.cost.detach(), post.u.detach(), post.Lambda_bar.detach(), post.R.detach(),
            post.row_failed, *[t.grad for t in leaves[:4]], lp_grad]


@pytest.mark.parametrize("N, budget", [(6, 12), (7, 5)])
def test_partly_masked_tiles_factor_only_the_admitted_pairs(N: int, budget: int,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    r"""Six receivers in one row block with source tiles of two (every tile straddles the causal diagonal: 36 pairs
    factored before), or seven in row blocks of five and two with one-source tiles: only the N(N + 1)/2 admitted
    pairs of each batch element are factored, bitwise the dense identity-masked tiles."""
    inputs = _pair_case(N=N)
    single = _posterior_and_grads(inputs)                              # one tile (no plan)
    monkeypatch.setattr(ma, "PAIR_TILE_BYTES", budget * 8 * 2 * 3 * 4)   # float64, batch 2, rank 2
    factored: List[int] = []
    real_cholesky = torch.linalg.cholesky_ex
    monkeypatch.setattr(torch.linalg, "cholesky_ex",
                        lambda W, **k: factored.append(W.shape[:-2].numel()) or real_cholesky(W, **k))
    with torch.no_grad():
        ma.exact_record_posterior(*inputs[:4], inputs[4], check=False)
    assert sum(factored) == 2 * N * (N + 1) // 2
    monkeypatch.setattr(torch.linalg, "cholesky_ex", real_cholesky)
    packed = _posterior_and_grads(inputs)
    plan = ma._tile_support_plan
    monkeypatch.setattr(ma, "_tile_support_plan", lambda *args: plan(*args)._replace(pairs={}))
    dense = _posterior_and_grads(inputs)
    for got, want in zip(packed, dense):
        assert torch.equal(got, want)
    for got, want in zip(packed, single):
        torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)
    masked = ~torch.isfinite(inputs[4])
    assert bool((packed[1][..., masked] == 0).all())                  # every masked pair's cost reads 0


def test_tile_plan_is_built_once_per_prior_version_and_view(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = _pair_case()
    monkeypatch.setattr(ma, "PAIR_TILE_BYTES", _TWO_SOURCE_TILES)
    builds: List[int] = []
    real = ma._build_tile_plan
    monkeypatch.setattr(ma, "_build_tile_plan", lambda *args: builds.append(1) or real(*args))
    lp = inputs[4].clone()
    first = ma.exact_record_posterior(*inputs[:4], lp)
    second = ma.exact_record_posterior(*inputs[:4], lp)
    assert len(builds) == 1
    assert all(torch.equal(x, y) for x, y in zip(first[:5], second[:5]))
    lp.add_(0.0)                                                       # an in-place write: a new version
    ma.exact_record_posterior(*inputs[:4], lp)
    assert len(builds) == 2
    # The readout slices one prior per head: one plan per head view, reused on the next call.
    gen = torch.Generator().manual_seed(10)
    xi = torch.randn(6, 4, generator=gen, dtype=F64)
    X = torch.randn(6, 4, 4, generator=gen, dtype=F64)
    C = X @ X.transpose(-1, -2) / 4 + 0.5 * torch.eye(4, dtype=F64)
    S, T = torch.randn(2, 2, 4, generator=gen, dtype=F64), torch.randn(2, 2, 4, generator=gen, dtype=F64)
    prior = lp.expand(2, 6, 6).clone()
    for _ in range(2):
        ma.minimal_agent_readout(xi, C, S, T, torch.full((2,), -0.5, dtype=F64), prior, "exact")
    assert len(builds) == 2 + 2


def test_the_body_checkpoint_reads_no_tile_mask_in_backward(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""Under ``predictive_covariance_checkpoint`` the body recomputes in backward; the cached plan serves it."""
    ops = make_relational_operators([2, 2], 0, attention_residual=False, predictive_readout="minimal_agent",
                                    minimal_agent_inference="exact", predictive_covariance="moment_matched",
                                    predictive_covariance_floor=1e-3, predictive_covariance_checkpoint=True)
    monkeypatch.setattr(ma, "PAIR_TILE_BYTES", 1)                     # one pair per tile: a multi-tile grid
    gen = torch.Generator().manual_seed(11)
    N = 4
    blocks = torch.eye(2) + 0.2 * torch.randn(1, N, 2, 2, 2, generator=gen)
    frame = CanonicalFrameContext(blocks, torch.linalg.inv(blocks), block_diagonal=True)
    mu = torch.randn(1, N, 4, generator=gen).requires_grad_(True)
    sigma = 0.5 + torch.rand(1, N, 4, generator=gen)
    lp = torch.zeros(N, N).masked_fill(torch.ones(N, N, dtype=torch.bool).triu(1), -math.inf)
    reads: List[str] = []
    tolist = torch.Tensor.tolist

    def spy(self: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        if torch._C._current_graph_task_id() != -1:
            reads.append("backward")
        return tolist(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "tolist", spy)
    mean, cov = ops.minimal_agent_update(mu, sigma, frame, lp, return_covariance=True, floor_anchor=sigma.detach())
    (mean.square().sum() + cov.sum()).backward()
    assert reads == [] and bool(torch.isfinite(mu.grad).all())


@pytest.mark.parametrize("inference", ["exact", "mean_field"])
@pytest.mark.parametrize("wide", [False, True])
def test_identity_frame_body_is_bitwise_the_generic_body(inference: str, wide: bool) -> None:
    r"""The frame-diagonal body reads (xi, diag c) through the identity frame: with ``identity_frame`` it forms no
    frame product, and its outputs and gradients equal the generic body's bit for bit (``wide``: the float64 recompute
    of a float32 call, which keeps the float32 identity frame)."""
    ops = make_relational_operators([2, 2], 0, attention_residual=False, predictive_readout="minimal_agent",
                                    minimal_agent_inference=inference, predictive_covariance="moment_matched",
                                    predictive_covariance_floor=1e-3, frame_diagonal=True)
    gen = torch.Generator().manual_seed(12)
    with torch.no_grad():
        ops.raw_score_q.copy_(torch.eye(4).reshape(2, 2, 4) + 0.3 * torch.randn(2, 2, 4, generator=gen))
        ops.T.copy_(torch.eye(4).reshape(2, 2, 4) + 0.3 * torch.randn(2, 2, 4, generator=gen))
    N = 4
    blocks = torch.eye(2) + 0.2 * torch.randn(1, N, 2, 2, 2, generator=gen)
    frame = identity_frame(CanonicalFrameContext(blocks, torch.linalg.inv(blocks), block_diagonal=True))
    dtype = F64 if wide else torch.float32
    mu = torch.randn(1, N, 4, generator=gen).to(dtype)
    c = (0.5 + torch.rand(1, N, 4, generator=gen)).to(dtype)
    prior = (0.5 + torch.rand(1, N, 4, generator=gen)).to(dtype)
    lp = torch.zeros(N, N, dtype=dtype).masked_fill(torch.ones(N, N, dtype=torch.bool).triu(1), -math.inf)
    floor = ops.floor_term(prior, c, frame, diagonal=True)
    w_mean, w_cov = torch.randn(1, N, 4, generator=gen).to(dtype), torch.randn(1, N, 4, generator=gen).to(dtype)
    outputs = []
    for shortcut in (False, True):
        ops.zero_grad(set_to_none=True)
        leaves = [mu.clone().requires_grad_(True), c.clone().requires_grad_(True)]
        mean, cov, _, status, receivers = ops._minimal_agent_body(leaves[0], leaves[1], frame, lp, True, False, floor,
                                                                  shortcut)
        ((w_mean * mean).sum() + (w_cov * cov).sum()).backward()
        outputs.append([mean.detach(), cov.detach(), status, receivers, *(t.grad for t in leaves),
                        *(p.grad for p in (ops.raw_score_q, ops.T, ops.channel_log_noise))])
    for got, want in zip(*outputs):
        assert torch.equal(got, want)
