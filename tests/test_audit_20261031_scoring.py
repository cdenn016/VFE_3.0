r"""Remediation of the October 31, 2026 audit, package ``scoring`` (the relational scorer and its moments).

* A137 (attention-scoring-1): the pair-KL expansions run through :class:`vfe3.relational_scoring._PairKLExpansion`,
  whose backward forms no pair grid; values and every gradient are bitwise those of the former
  :class:`vfe3.relational_scoring._IEEEFP32Recompute` route, standalone and on a tiny live-route model.
* A149 (attention-scoring-9): the diagonal routes' certificate flags come from one ``aminmax`` per tensor, with the
  same verdicts as the composite ``isfinite`` chains.
* A143 (triton-kernels-1): the diagonal-moments forward kernel's 1-D row-fastest grid matches a copy of the former
  2-D grid bitwise under the Triton interpreter.
* A136 (relational-operators-1), A139 (perf-live-path-1): opt-in type-level moments and the shared canonical mean,
  with the frame-sized additions of backward counted as the audit counted them.
* A316 (numerics-5): the opt-in rotation-aware shift of the diagonal pair-KL expansion.
* A012, A013, A179 (gap-certified-factorization-routes-1, -2, -5): certified factor and noisy-key routes and the work
  of a recovery, including the token flags of the factor tiles, which the committed code read from the head axis.
* A311 (attention-scoring-8), A353 (attention-scoring-10): the tile-loop record and the mixture-mean oracle.

Tiny CPU cases only; configurations come from :mod:`tests.profiles`; the CUDA parity test skips unless the CUDA lane
runs it.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import vfe3.relational_scoring as rs
from tests.profiles import test_config
from vfe3.config import VFE3Config
from vfe3.runtime import ieee_fp32_matmul

REPO = Path(__file__).resolve().parents[1]
_NO_TRITON = 77
F32, F64 = torch.float32, torch.float64

requires_cuda = pytest.mark.skipif(
    os.environ.get("VFE3_TEST_DEVICE") != "cuda" or not torch.cuda.is_available(),
    reason="requires the CUDA lane (VFE3_TEST_DEVICE=cuda) and a CUDA device")


@pytest.fixture(autouse=True)
def _restore_scorer_globals() -> Any:
    previous = (rs.projected_condition_recovery(), rs.projected_factor_route(), rs.diagonal_moments_kernel(),
                rs.projection_tile_bytes())
    yield
    rs.set_projected_condition_recovery(previous[0])
    rs.set_projected_factor_route(previous[1])
    rs.set_diagonal_moments_kernel(previous[2])
    rs.set_projection_tile_bytes(previous[3])


# The live route's switches (docs/audits/ultradeep-2026-10-31/live-config.json) at tiny scale, from the test baseline:
# noisy-key routing on the diagonal comparison route, passthrough covariance seams, kernel readout, no frame dropout.
_LIVE_ROUTE: Dict[str, Any] = dict(
    vocab_size=7, embed_dim=8, n_heads=2, max_seq_len=6, batch_size=2, n_layers=2, n_e_steps=1,
    attention_mode="projected_predictive", predictive_readout="kernel", predictive_covariance="passthrough",
    relational_routing="noisy_key", rope_mode="comparison_invariant", rope_base=100.0, rope_fraction=1.0,
    routing_kl_max_reference="query_shifted", family="gaussian_diagonal", projected_factorization="gram",
    pos_phi="none", encode_mode="per_token", content_link_chart="exponential", attention_residual=True,
    norm_type_block="layernorm", block_norm_covariance="passthrough", use_block_mlp=True, block_mlp_mode="coordinate",
    block_mlp_covariance="passthrough", block_mlp_tie_layers=True, block_mlp_expansion=2,
    beta_attention_prior="causal_alibi", decode_mode="diagonal_chunked", decode_tau=0.1, decode_unigram_prior=True,
    gauge_group="block_glk", gauge_tying="per_head", kl_max=16.0, skip_belief_sigma_update=False,
    relational_receiver_measurement="inverse", relational_receiver_update="closed_form",
    decode_av_precision="fp32", decode_ranking_fp64_escalation=False, phi_scale=0.3, sigma_init=1.5,
    phi_type_dropout=0.0,
)
# Repeated token types, so the per-type shortcuts have rows to share.
_TOKENS = torch.tensor([[1, 2, 1, 3, 2, 1], [2, 2, 4, 1, 5, 1]])
_TARGETS = torch.tensor([[2, 1, 3, 2, 1, 0], [2, 4, 1, 5, 1, 6]])


def _live_config(**overrides: Any) -> VFE3Config:
    return test_config(**{**_LIVE_ROUTE, **overrides})


def _model_step(cfg: VFE3Config, *, dtype: torch.dtype = F32, train: bool = True,
                tokens: torch.Tensor = _TOKENS, targets: torch.Tensor = _TARGETS
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    r"""Loss and every parameter gradient of one forward and backward from a fixed initialization."""
    from vfe3.model.model import VFEModel
    torch.manual_seed(0)
    model = VFEModel(cfg).to(dtype)
    model.train(train)
    _, loss, _ = model(tokens, targets)
    loss.backward()
    return loss.detach(), {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def _equal_steps(a: Tuple[torch.Tensor, Dict[str, torch.Tensor]],
                 b: Tuple[torch.Tensor, Dict[str, torch.Tensor]]) -> bool:
    return torch.equal(a[0], b[0]) and a[1].keys() == b[1].keys() and all(torch.equal(a[1][n], b[1][n]) for n in a[1])


def _max_relative(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> float:
    return max(float((a[n] - b[n]).norm() / b[n].norm().clamp_min(1e-300)) for n in b if float(b[n].norm()) > 0)


# ------------------------------------------------------------------------------------------------ A137

@ieee_fp32_matmul()
def _former_diagonal_expansion(aq: torch.Tensor, ak: torch.Tensor, gq: torch.Tensor, gk: torch.Tensor,
                               logdet_q: torch.Tensor) -> torch.Tensor:
    r"""``_diagonal_kl_expansion`` as committed at 5bb0a57, verbatim (the body the former recompute differentiated)."""
    r = aq.shape[-1]
    c = ak[..., :1, :].detach()
    alpha, b = aq - c, ak - c
    inverse = gk.reciprocal()
    scaled = b * inverse
    query = torch.cat((gq + alpha * alpha, alpha), dim=-1)
    key = torch.cat((inverse, -2 * scaled), dim=-1)
    cross = query @ key.transpose(-1, -2)
    key_constant = (b * scaled).sum(-1) + gk.log().sum(-1)
    return .5 * (cross + key_constant.unsqueeze(-2) - r - logdet_q.unsqueeze(-1))


@ieee_fp32_matmul()
def _former_dense_expansion(aq: torch.Tensor, ak: torch.Tensor, q_cov: torch.Tensor, k_prec: torch.Tensor,
                            logdet_q: torch.Tensor, logdet_k: torch.Tensor) -> torch.Tensor:
    r"""``_projected_kl_expansion`` as committed at 5bb0a57, verbatim."""
    r = aq.shape[-1]
    c = ak[..., :1, :].detach()
    alpha, b = aq - c, ak - c
    Pb = (k_prec @ b.unsqueeze(-1)).squeeze(-1)
    PTb = (k_prec.transpose(-1, -2) @ b.unsqueeze(-1)).squeeze(-1)
    second = q_cov + alpha.unsqueeze(-1) * alpha.unsqueeze(-2)
    query = torch.cat((second.flatten(-2), alpha), dim=-1)
    key = torch.cat((k_prec.flatten(-2), -(Pb + PTb)), dim=-1)
    cross = query @ key.transpose(-1, -2)
    key_constant = (b * Pb).sum(-1) + logdet_k
    return .5 * (cross + key_constant.unsqueeze(-2) - r - logdet_q.unsqueeze(-1))


def _former_pair_kl_autograd(operands: Any, r: int, *inputs: torch.Tensor) -> torch.Tensor:
    r"""The pre-change route: the former expansion bodies through :class:`_IEEEFP32Recompute`."""
    if operands is rs._diagonal_kl_operands:
        return rs._ieee_fp32_autograd(_former_diagonal_expansion, *inputs)
    if operands is rs._projected_kl_operands:
        aq, ak, q_cov, k_prec, logdet_k, logdet_q = inputs
        return rs._ieee_fp32_autograd(_former_dense_expansion, aq, ak, q_cov, k_prec, logdet_q, logdet_k)
    raise AssertionError("unexpected pair-KL operands")


class _GridOps(TorchDispatchMode):
    r"""Counts dispatched ops whose first output has ``numel`` pair-grid elements."""

    def __init__(self, grid: int) -> None:
        super().__init__()
        self.grid, self.ops = grid, Counter()

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        out = func(*args, **(kwargs or {}))
        first = out[0] if isinstance(out, (tuple, list)) and out else out
        if isinstance(first, torch.Tensor) and first.numel() == self.grid:
            self.ops[func.overloadpacket.__name__] += 1
        return out


def _diagonal_inputs(dtype: torch.dtype, differentiable_logdet: bool) -> List[torch.Tensor]:
    g = torch.Generator().manual_seed(3)
    B, H, N, r = 2, 3, 7, 4
    aq = torch.randn(B, H, N, r, generator=g, dtype=dtype)
    ak = torch.randn(B, H, N, r, generator=g, dtype=dtype) + 2.0
    gq = .5 + torch.rand(B, H, N, r, generator=g, dtype=dtype)
    gk = .5 + torch.rand(B, H, N, r, generator=g, dtype=dtype)
    logdet = (torch.randn(B, H, N, generator=g, dtype=dtype) if differentiable_logdet
              else torch.full((B, H, N), -float(r), dtype=dtype))
    return [aq, ak, gq, gk, logdet]


def _dense_inputs(dtype: torch.dtype) -> List[torch.Tensor]:
    g = torch.Generator().manual_seed(5)
    B, H, N, r = 2, 2, 6, 3
    aq = torch.randn(B, H, N, r, generator=g, dtype=dtype)
    ak = torch.randn(B, H, N, r, generator=g, dtype=dtype)
    A = torch.randn(B, H, N, r, r, generator=g, dtype=dtype)
    q_cov = A @ A.transpose(-1, -2) + torch.eye(r, dtype=dtype)
    C = torch.randn(B, H, N, r, r, generator=g, dtype=dtype)
    k_prec = C @ C.transpose(-1, -2) + torch.eye(r, dtype=dtype)
    return [aq, ak, q_cov, k_prec, torch.randn(B, H, N, generator=g, dtype=dtype),
            torch.randn(B, H, N, generator=g, dtype=dtype)]


@pytest.mark.parametrize("dtype", [F32, F64])
@pytest.mark.parametrize("differentiable_logdet", [True, False])
def test_diagonal_expansion_function_is_bitwise_the_former_recompute(dtype: torch.dtype,
                                                                     differentiable_logdet: bool) -> None:
    inputs = _diagonal_inputs(dtype, differentiable_logdet)
    needs = [True, True, True, True, differentiable_logdet]
    leaves_a = [x.clone().requires_grad_(need) for x, need in zip(inputs, needs)]
    leaves_b = [x.clone().requires_grad_(need) for x, need in zip(inputs, needs)]
    old = rs._ieee_fp32_autograd(_former_diagonal_expansion, *leaves_a)
    new = rs._pair_kl_autograd(rs._diagonal_kl_operands, inputs[0].shape[-1], *leaves_b)
    assert torch.equal(old, new)
    cot = torch.randn(old.shape, generator=torch.Generator().manual_seed(9), dtype=dtype)
    grid = old.numel()
    counter_old, counter_new = _GridOps(grid), _GridOps(grid)
    with counter_old:
        g_old = torch.autograd.grad((old * cot).sum(), [x for x in leaves_a if x.requires_grad])
    with counter_new:
        g_new = torch.autograd.grad((new * cot).sum(), [x for x in leaves_b if x.requires_grad])
    assert all(torch.equal(a, b) for a, b in zip(g_old, g_new))
    # The former backward rebuilt the pair grid (its GEMM and epilogue); the new one forms no pair-grid GEMM.
    assert counter_old.ops["bmm"] >= 1 and counter_new.ops["bmm"] == 0
    assert sum(counter_new.ops.values()) < sum(counter_old.ops.values())


@pytest.mark.parametrize("dtype", [F32, F64])
def test_dense_expansion_function_is_bitwise_the_former_recompute(dtype: torch.dtype) -> None:
    inputs = _dense_inputs(dtype)
    leaves_a = [x.clone().requires_grad_(True) for x in inputs]
    leaves_b = [x.clone().requires_grad_(True) for x in inputs]
    old = rs._ieee_fp32_autograd(_former_dense_expansion, *leaves_a)
    new = rs._projected_kl_gemm(*leaves_b)
    assert torch.equal(old, new)
    cot = torch.randn(old.shape, generator=torch.Generator().manual_seed(2), dtype=dtype)
    g_old = torch.autograd.grad((old * cot).sum(), leaves_a)
    g_new = torch.autograd.grad((new * cot).sum(), leaves_b)
    assert all(torch.equal(a, b) for a, b in zip(g_old, g_new))


@pytest.mark.parametrize("overrides", [
    {},                                                                                    # the live diagonal route
    dict(relational_routing="gaussian_kl"),
    dict(rope_mode="none", relational_routing="gaussian_kl"),                              # the dense Gram route
    dict(rope_mode="none"),                                                                # dense noisy key
])
def test_model_loss_and_every_gradient_are_bitwise_the_former_route(overrides: Dict[str, Any],
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _live_config(**overrides)
    new = _model_step(cfg)
    monkeypatch.setattr(rs, "_pair_kl_autograd", _former_pair_kl_autograd)
    old = _model_step(cfg)
    assert new[1] and _equal_steps(new, old)


@requires_cuda
def test_cuda_expansion_function_is_bitwise_the_former_recompute_under_tf32_policy() -> None:
    r"""The adoption gate of A137 on CUDA: under ``fp32_matmul_precision='high'`` the new backward's GEMMs, pinned to
    IEEE fp32 on the same views, give the former recompute's gradients bit for bit."""
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        for differentiable_logdet in (True, False):
            inputs = [x.cuda() for x in _diagonal_inputs(F32, differentiable_logdet)]
            needs = [True, True, True, True, differentiable_logdet]
            leaves_a = [x.clone().requires_grad_(need) for x, need in zip(inputs, needs)]
            leaves_b = [x.clone().requires_grad_(need) for x, need in zip(inputs, needs)]
            old = rs._ieee_fp32_autograd(_former_diagonal_expansion, *leaves_a)
            new = rs._pair_kl_autograd(rs._diagonal_kl_operands, inputs[0].shape[-1], *leaves_b)
            cot = torch.randn(old.shape, device="cuda", generator=torch.Generator("cuda").manual_seed(9))
            g_old = torch.autograd.grad((old * cot).sum(), [x for x in leaves_a if x.requires_grad])
            g_new = torch.autograd.grad((new * cot).sum(), [x for x in leaves_b if x.requires_grad])
            assert torch.equal(old, new) and all(torch.equal(a, b) for a, b in zip(g_old, g_new))
        inputs = [x.cuda() for x in _dense_inputs(F32)]
        leaves_a = [x.clone().requires_grad_(True) for x in inputs]
        leaves_b = [x.clone().requires_grad_(True) for x in inputs]
        old = rs._ieee_fp32_autograd(_former_dense_expansion, *leaves_a)
        new = rs._projected_kl_gemm(*leaves_b)
        cot = torch.randn(old.shape, device="cuda", generator=torch.Generator("cuda").manual_seed(2))
        g_old = torch.autograd.grad((old * cot).sum(), leaves_a)
        g_new = torch.autograd.grad((new * cot).sum(), leaves_b)
        assert torch.equal(old, new) and all(torch.equal(a, b) for a, b in zip(g_old, g_new))
    finally:
        torch.set_float32_matmul_precision(previous)


# ------------------------------------------------------------------------------------------------ A149

def _composite_flags(aq: torch.Tensor, ak: torch.Tensor, gq: torch.Tensor, gk: torch.Tensor) -> torch.Tensor:
    return torch.stack((torch.isfinite(aq).all() & torch.isfinite(ak).all(), torch.isfinite(gq).all(), (gq > 0).all(),
                        torch.isfinite(gk).all(), (gk > 0).all()))


@pytest.mark.parametrize("dtype", [F32, F64])
def test_aminmax_flags_give_the_composite_verdicts(dtype: torch.dtype) -> None:
    g = torch.Generator().manual_seed(0)
    base = [torch.randn(2, 3, 5, 4, generator=g, dtype=dtype), torch.randn(2, 3, 5, 4, generator=g, dtype=dtype),
            .5 + torch.rand(2, 3, 5, 4, generator=g, dtype=dtype), .5 + torch.rand(2, 3, 5, 4, generator=g, dtype=dtype)]
    cases = [base]
    for which in range(4):
        for bad in (float("nan"), float("inf"), float("-inf"), 0.0, -1.0):
            tensors = [x.clone() for x in base]
            tensors[which][1, 2, 3, 0] = bad
            cases.append(tensors)
    cases.append([torch.empty(0, 3, 5, 4, dtype=dtype)] * 4)
    for tensors in cases:
        assert torch.equal(rs._diagonal_route_flags(*tensors), _composite_flags(*tensors))

    class _FullPasses(TorchDispatchMode):
        r"""Counts dispatched ops that read a whole input tensor (a pass over the means or variances)."""

        def __init__(self, numel: int) -> None:
            super().__init__()
            self.numel, self.count = numel, 0

        def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
            if any(isinstance(a, torch.Tensor) and a.numel() == self.numel for a in args):
                self.count += 1
            return func(*args, **(kwargs or {}))

    counter_old, counter_new = _FullPasses(base[0].numel()), _FullPasses(base[0].numel())
    with counter_old:
        _composite_flags(*base)
    with counter_new:
        rs._diagonal_route_flags(*base)
    # One aminmax per tensor, against at least an elementwise kernel and a reduction per verdict (24 dispatched ops
    # here: isfinite itself decomposes).
    assert counter_new.count == 4 and counter_old.count >= 12


def test_diagonal_route_still_raises_its_verdicts() -> None:
    g = torch.Generator().manual_seed(1)
    a = torch.randn(1, 4, 2, 3, generator=g)
    good = .5 + torch.rand(1, 4, 2, 3, generator=g)
    bad = good.clone()
    bad[0, 1, 0, 2] = -1.0
    with pytest.raises(ValueError, match="not positive"):
        rs.projected_gaussian_kl(a, good, a, bad, diagonal_covariance=True)
    nan = good.clone()
    nan[0, 2, 1, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        rs.projected_noisy_query_cost(a, nan, a, good, diagonal_covariance=True)


# ------------------------------------------------------------------------------------------------ A143

_GRID_SCRIPT = """
import json, sys
try:
    import triton
    import triton.language as tl
except Exception as exc:
    print("triton import failed:", exc)
    sys.exit(77)
import torch
from vfe3.kernels import _diagonal_moments_triton as tk
from vfe3.kernels import diagonal_moments as dm
assert tk.INTERPRETED
torch.set_num_threads(2)


@triton.jit
def former_fwd(S, U, ROOT, G, T, ROWS, K, M, d, BR: tl.constexpr, BD: tl.constexpr, TB: tl.constexpr):
    # The forward kernel with the former 2-D grid: program_id(0) the token group, program_id(1) the row tile.
    t0 = tl.program_id(0) * TB
    rows = tl.program_id(1) * BR + tl.arange(0, BR)
    row_ok = rows < ROWS
    offs_c = tl.arange(0, BD)
    c_ok = offs_c < d
    cols = tl.arange(0, TB * BD)
    acc = tl.zeros([BR, TB], S.dtype.element_ty)
    for m in range(0, M):
        s_tile = tk._load_maps(S, rows, row_ok, m, offs_c, c_ok, K, d)
        u_tile = tk._load_frames(U, t0, m, offs_c, c_ok, cols, T, M, d, BD)
        root, _, _, _ = tk._load_roots(ROOT, t0, m, cols, T, K, d, BD)
        x = tl.dot(s_tile, u_tile, input_precision="ieee") * root[None, :]
        acc += tl.sum(tl.reshape(x * x, (BR, TB, BD)), 2)
    toks = t0 + tl.arange(0, TB)
    ptr = G + toks.to(tl.int64)[None, :] * ROWS + rows[:, None]
    tl.store(ptr, acc, mask=row_ok[:, None] & (toks < T)[None, :])


out = {}
for dtype in (torch.float32, torch.float64):
    for T, M, d, ROWS in ((11, 3, 5, 37), (9, 2, 16, 70), (5, 1, 32, 33)):
        g = torch.Generator().manual_seed(T + ROWS)
        K = M * d
        S = torch.randn(ROWS, K, generator=g).to(dtype)
        U = (torch.eye(d) + 0.3 * torch.randn(T, M, d, d, generator=g)).to(dtype)
        root = (0.2 + torch.rand(T, K, generator=g)).sqrt().to(dtype)
        c = tk.launch_config(d, dtype == torch.float64)
        ref = torch.empty(T, ROWS, dtype=dtype)
        former_fwd[(-(-T // c["TB"]), -(-ROWS // c["BR_FWD"]))](S, U, root, ref, T, ROWS, K, M, d,
                                                                BR=c["BR_FWD"], BD=c["BD"], TB=c["TB"])
        new = dm.diagonal_root_square_sums(S, U, root)
        out[f"{dtype}_{T}_{M}_{d}_{ROWS}"] = bool(torch.equal(ref, new))
print(json.dumps(out))
"""


def _run_interpreter(script: str) -> Any:
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES="-1", VFE3_TEST_DEVICE="cpu", PYTHONPATH=str(REPO), KMP_DUPLICATE_LIB_OK="TRUE",
               TRITON_INTERPRET="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    proc = subprocess.run([sys.executable, "-c", textwrap.dedent(script)], cwd=REPO, env=env, capture_output=True,
                          text=True, timeout=900)
    if proc.returncode == _NO_TRITON:
        pytest.skip(f"triton unavailable in {sys.executable}")
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_forward_kernel_row_fastest_grid_is_bitwise_the_former_grid() -> None:
    results = _run_interpreter(_GRID_SCRIPT)
    assert results and all(results.values()), results


# ------------------------------------------------------------------------------------------------ A136

class _MomentsSpy:
    r"""Records, per split-moments call, whether ``type_rows`` and ``canonical_mean`` were set and the variance rows."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls: List[Dict[str, Any]] = []
        real_split, real_sums = rs._split_diagonal_moments, rs._diagonal_square_sums

        def split(mu_h, Sigma_h, inverse_blocks, S, dtype, batch, canonical_mean, type_rows):  # type: ignore[no-untyped-def]
            self.calls.append({"type_rows": type_rows is not None, "canonical_mean": canonical_mean is not None,
                               "rows": []})
            return real_split(mu_h, Sigma_h, inverse_blocks, S, dtype, batch, canonical_mean, type_rows)

        def sums(Sigma_h, inverse_blocks, maps, dtype, batch):  # type: ignore[no-untyped-def]
            if self.calls:
                self.calls[-1]["rows"].append(math.prod(Sigma_h.shape[:-2]))
            return real_sums(Sigma_h, inverse_blocks, maps, dtype, batch)

        monkeypatch.setattr(rs, "_split_diagonal_moments", split)
        monkeypatch.setattr(rs, "_diagonal_square_sums", sums)


def test_type_moments_evaluate_the_distinct_types_with_the_same_costs(monkeypatch: pytest.MonkeyPatch) -> None:
    base = _model_step(_live_config())
    spy = _MomentsSpy(monkeypatch)
    on = _model_step(_live_config(relational_type_moments=True))
    distinct = int(torch.unique(_TOKENS).numel())
    assert spy.calls and all(call["type_rows"] for call in spy.calls)
    assert all(call["rows"] == [distinct] for call in spy.calls)               # 5 types, not 12 occurrences
    assert torch.equal(on[0], base[0])                                           # the costs and the loss
    assert _max_relative(on[1], base[1]) < 1e-5                                  # gradients at rounding level


def test_type_moments_gradients_agree_in_float64() -> None:
    base = _model_step(_live_config(), dtype=F64)
    on = _model_step(_live_config(relational_type_moments=True), dtype=F64)
    assert float((on[0] - base[0]).abs()) <= 1e-13 * float(base[0].abs())
    assert _max_relative(on[1], base[1]) < 1e-12


@pytest.mark.parametrize("overrides,train,declared", [
    (dict(phi_type_dropout=0.1), True, False),                       # training frame dropout: per-sequence frames
    (dict(phi_type_dropout=0.1), False, True),                       # an evaluation forward draws no dropout
    (dict(pos_phi="learned"), True, False),                          # positional frames
    (dict(relational_type_moments=False), True, False),
])
def test_type_moments_gate_refuses_forwards_that_are_not_type_determined(
        overrides: Dict[str, Any], train: bool, declared: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    spy = _MomentsSpy(monkeypatch)
    _model_step(_live_config(**{"relational_type_moments": True, **overrides}), train=train)
    assert any(call["type_rows"] for call in spy.calls) is declared


def test_public_belief_forwards_make_no_declaration(monkeypatch: pytest.MonkeyPatch) -> None:
    from vfe3.model.model import VFEModel
    torch.manual_seed(0)
    model = VFEModel(_live_config(relational_type_moments=True))
    spy = _MomentsSpy(monkeypatch)
    with torch.no_grad():
        model.forward_beliefs(_TOKENS)              # public beliefs keep occurrence rows: no content-row reuse
    assert not any(call["type_rows"] for call in spy.calls)


def test_declaration_holds_for_one_covariance_version_and_one_frame() -> None:
    from vfe3.contracts import CanonicalFrameContext, mark_gathered_frame_index, mark_gathered_frame_pair
    g = torch.Generator().manual_seed(0)
    unique = torch.randn(3, 2, 4, 4, generator=g) + 3 * torch.eye(4)               # (U, H, d, d)
    inverse = torch.linalg.inv(unique)
    index = torch.tensor([0, 1, 0, 2, 1, 0])
    forward_occ = unique.index_select(0, index).reshape(1, 6, 2, 4, 4)              # (B, N, H, d, d)
    inverse_occ = inverse.index_select(0, index).reshape(1, 6, 2, 4, 4)
    sigma = torch.rand(3, 8, generator=g).index_select(0, index).reshape(1, 6, 8) + 0.5
    unmarked = CanonicalFrameContext(forward=forward_occ, inverse=inverse_occ, block_diagonal=True)
    assert not rs.declare_type_rows(sigma, unmarked)                                 # no row record on the frame
    mark_gathered_frame_pair(forward_occ, inverse_occ, unique, inverse)
    mark_gathered_frame_index(forward_occ, inverse_occ, index)
    frame = CanonicalFrameContext(forward=forward_occ, inverse=inverse_occ, block_diagonal=True)
    assert rs.declare_type_rows(sigma, frame)
    rows = rs.type_rows(sigma, frame)
    assert rows is not None and rows.index is index and rows.inverse is inverse
    assert rows.first.tolist() == [0, 1, 3] and rows.lengths.tolist() == [3, 2, 1]
    assert rows.order.tolist() == [0, 2, 5, 1, 4, 3]
    other = CanonicalFrameContext(forward=forward_occ.clone(), inverse=inverse_occ.clone(), block_diagonal=True)
    assert rs.type_rows(sigma, other) is None                                        # another frame
    assert rs.type_rows(sigma.clone(), frame) is None                                # another covariance tensor
    sigma.mul_(1.0)
    assert rs.type_rows(sigma, frame) is None                                        # an in-place write


def test_type_row_gather_backward_is_a_deterministic_segment_sum() -> None:
    g = torch.Generator().manual_seed(4)
    index = torch.tensor([2, 0, 1, 2, 2, 0, 3, 1])
    order = torch.argsort(index, stable=True)
    lengths = torch.zeros(4, dtype=torch.long).scatter_add_(0, index, torch.ones_like(index))
    rows = torch.randn(4, 5, generator=g, dtype=F64, requires_grad=True)
    cot = torch.randn(8, 5, generator=g, dtype=F64)
    out = rs._TypeRowGather.apply(rows, index, order, lengths)
    assert torch.equal(out, rows.index_select(0, index))
    grad, = torch.autograd.grad((out * cot).sum(), rows)
    assert torch.equal(grad, torch.zeros(4, 5, dtype=F64).index_add_(0, index, cot))
    assert torch.autograd.gradcheck(lambda x: rs._TypeRowGather.apply(x, index, order, lengths), (rows,))


def test_type_level_moments_match_the_occurrence_route_and_finite_differences() -> None:
    r"""``projected_moments(type_rows=...)``: the variances of the distinct rows gathered to the occurrences equal the
    per-occurrence route bitwise, and their float64 gradients in the variances, the inverse frames and the maps pass
    central differences."""
    g = torch.Generator().manual_seed(7)
    H, d, r, K = 2, 2, 2, 4
    index = torch.tensor([0, 1, 0, 2, 1])
    order = torch.argsort(index, stable=True)
    lengths = torch.zeros(3, dtype=torch.long).scatter_add_(0, index, torch.ones_like(index))
    first = torch.tensor([0, 1, 3])
    mu_h = torch.randn(1, 5, K, generator=g, dtype=F64).unsqueeze(-2).expand(1, 5, H, K)

    def moments(sigma_rows: torch.Tensor, inverse_rows: torch.Tensor, S: torch.Tensor,
                typed: bool) -> torch.Tensor:                                          # (R, 1, 5, H, r)
        sigma_h = sigma_rows.index_select(0, index).reshape(1, 5, K).unsqueeze(-2).expand(1, 5, H, K)
        inverse = inverse_rows.index_select(0, index).reshape(1, 5, H, d, d)
        rows = rs.TypeRows(index=index, first=first, order=order, lengths=lengths, inverse=inverse_rows)
        return rs.projected_moments(mu_h, sigma_h, inverse, S, diagonal_only=True,
                                    type_rows=rows if typed else None)[1]

    sigma_rows = (0.5 + torch.rand(3, K, generator=g, dtype=F64)).requires_grad_(True)
    inverse_rows = (torch.eye(d, dtype=F64) + 0.3 * torch.randn(3, H, d, d, generator=g, dtype=F64)).requires_grad_(True)
    S = torch.randn(2, H, r, K, generator=g, dtype=F64).requires_grad_(True)
    with torch.no_grad():
        assert torch.equal(moments(sigma_rows, inverse_rows, S, True), moments(sigma_rows, inverse_rows, S, False))
    assert torch.autograd.gradcheck(lambda *x: moments(*x, True), (sigma_rows, inverse_rows, S))


# ------------------------------------------------------------------------------------------------ A139

def _frame_matvecs(cfg: VFE3Config) -> int:
    r"""Products :math:`U_i^{-1}\mu_i` formed in one forward and backward: :meth:`RelationalOperators.canonical_mean`
    calls and the scorer's projected means read through a frame."""
    from vfe3.model.relational_operators import RelationalOperators
    count = [0]
    real_root_mean, real_canonical = rs._projected_root_mean, RelationalOperators.canonical_mean

    def root_mean(mu: torch.Tensor, inverse: Any, S: torch.Tensor) -> torch.Tensor:
        count[0] += inverse is not None
        return real_root_mean(mu, inverse, S)

    def canonical(self: Any, mu: torch.Tensor, frame: Any) -> torch.Tensor:
        count[0] += 1
        return real_canonical(self, mu, frame)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(rs, "_projected_root_mean", root_mean)
        patch.setattr(RelationalOperators, "canonical_mean", canonical)
        _model_step(cfg)
    return count[0]


def test_shared_canonical_mean_keeps_the_costs_and_forms_one_frame_product_per_layer(
        monkeypatch: pytest.MonkeyPatch) -> None:
    base = _model_step(_live_config())
    spy = _MomentsSpy(monkeypatch)
    on = _model_step(_live_config(relational_shared_canonical_mean=True))
    assert [call["canonical_mean"] for call in spy.calls] == [True] * _LIVE_ROUTE["n_layers"]
    assert torch.equal(on[0], base[0])
    assert _max_relative(on[1], base[1]) < 1e-5
    on64 = _model_step(_live_config(relational_shared_canonical_mean=True), dtype=F64)
    base64 = _model_step(_live_config(), dtype=F64)
    assert _max_relative(on64[1], base64[1]) < 1e-12
    monkeypatch.undo()
    saved = _frame_matvecs(_live_config()) - _frame_matvecs(_live_config(relational_shared_canonical_mean=True))
    assert saved == _LIVE_ROUTE["n_layers"]


class _FrameAdds(TorchDispatchMode):
    r"""Counts ``add`` dispatches whose output has the occurrence frame's shape (B, N, H, d, d)."""

    def __init__(self, frame: Tuple[int, ...]) -> None:
        super().__init__()
        self.frame, self.count = frame, 0

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        out = func(*args, **(kwargs or {}))
        if func.__name__.startswith("add") and isinstance(out, torch.Tensor) and tuple(out.shape) == self.frame:
            self.count += 1
        return out


def _frame_sized_additions(n_layers: int, **overrides: Any) -> int:
    r"""Frame-sized additions in one backward, on the shape of the audit's probe
    perf-live-path_frame_grad_accumulation.py (B = 2, N = 11, H = 3, K = 12); phi_scale = 0.3 keeps the frames off
    the identity."""
    from vfe3.contracts import deferred_host_checks
    from vfe3.model.model import VFEModel
    B, N, H, K, V = 2, 11, 3, 12, 9
    torch.manual_seed(0)
    model = VFEModel(_live_config(embed_dim=K, n_heads=H, max_seq_len=N, batch_size=B, vocab_size=V,
                                  n_layers=n_layers, **overrides))
    model.train()
    g = torch.Generator().manual_seed(7)
    x, y = torch.randint(0, V, (B, N), generator=g), torch.randint(0, V, (B, N), generator=g)
    with deferred_host_checks():
        _, loss, _ = model(x, y)
    counter = _FrameAdds((B, N, H, K // H, K // H))
    with counter:
        loss.backward()
    return counter.count


@pytest.mark.parametrize("n_layers", [2, 3])
def test_opt_in_shortcuts_remove_frame_sized_additions(n_layers: int) -> None:
    r"""4L - 2 frame-sized additions on the default route (the audit's count), 3L - 2 with either shortcut (the shared
    mean removes the readout's second U^{-1} mu, the type moments move the moments' dU to the distinct rows), and
    2L - 2 with both."""
    L = n_layers
    assert _frame_sized_additions(L) == 4 * L - 2
    assert _frame_sized_additions(L, relational_shared_canonical_mean=True) == 3 * L - 2
    assert _frame_sized_additions(L, relational_type_moments=True) == 3 * L - 2
    assert _frame_sized_additions(L, relational_type_moments=True, relational_shared_canonical_mean=True) == 2 * L - 2


def test_shared_canonical_mean_is_inert_off_the_diagonal_route() -> None:
    cfg = _live_config(rope_mode="none", relational_shared_canonical_mean=True)
    assert "relational_shared_canonical_mean" in cfg.inactive_fields()
    assert _equal_steps(_model_step(cfg), _model_step(_live_config(rope_mode="none")))


# ------------------------------------------------------------------------------------------------ A316

def _rotated_case(offset: float, first_scale: float, seed: int = 0) -> Tuple[torch.Tensor, ...]:
    r"""Comparison means rotated by RoPE (base 100, every coordinate rotating), with a common offset and a scaled first
    token, and unit-scale variances, as the diagonal route scores them; float32."""
    from vfe3.geometry.rope import build_rope_rotation
    g = torch.Generator().manual_seed(seed)
    N, H, r = 64, 1, 8
    a = torch.randn(1, N, H, r, generator=g, dtype=F64) + offset
    a[:, 0] *= first_scale
    R = build_rope_rotation(torch.arange(N), [r], base=100.0, fraction=1.0, dtype=F64)            # (N, r, r)
    a = torch.einsum("nst,znht->znhs", R, a)
    gq = 0.5 + torch.rand(1, N, H, r, generator=g, dtype=F64)
    gk = 0.5 + torch.rand(1, N, H, r, generator=g, dtype=F64)
    return tuple(x.float() for x in (a, gq, gk))


def _direct_kl64(a: torch.Tensor, gq: torch.Tensor, gk: torch.Tensor) -> torch.Tensor:   # (1, H, N, N)
    aq, g_q, g_k = (x.double().movedim(-2, -3) for x in (a, gq, gk))                      # (1, H, N, r)
    delta = aq.unsqueeze(-2) - aq.unsqueeze(-3)
    return .5 * ((g_q.unsqueeze(-2) / g_k.unsqueeze(-3) + delta.square() / g_k.unsqueeze(-3)
                  + g_k.unsqueeze(-3).log() - g_q.unsqueeze(-2).log()).sum(-1) - a.shape[-1])


def test_expansion_shift_is_zero_in_the_rotating_coordinates() -> None:
    ak = torch.arange(10.0).reshape(1, 1, 2, 5)
    assert torch.equal(rs._expansion_shift(ak, 0), ak[..., :1, :])
    assert rs._expansion_shift(ak, 4).tolist() == [[[[0.0, 0.0, 0.0, 0.0, 4.0]]]]
    a, gq, gk = _rotated_case(0.0, 1.0)
    for bad in (9, -1, True):
        with pytest.raises(ValueError, match="rotating_coordinates"):
            rs.projected_gaussian_kl(a, gq, a, gk, diagonal_covariance=True, rotating_coordinates=bad)


@pytest.mark.parametrize("offset,first_scale", [(10.0, 1.0), (0.0, 30.0), (30.0, 1.0)])
def test_rotation_aware_shift_is_no_worse_than_the_first_key_under_rope(offset: float, first_scale: float) -> None:
    a, gq, gk = _rotated_case(offset, first_scale)
    reference = _direct_kl64(a, gq, gk)
    relevant = reference - reference.amin(-1, keepdim=True) < 30.0
    errors, estimates = {}, {}
    for m in (0, 8):
        cost = rs.projected_gaussian_kl(a, gq, a, gk, diagonal_covariance=True, rotating_coordinates=m).double()
        errors[m] = float((cost - reference).abs()[relevant].max())
        aq, g_k = a.movedim(-2, -3), gk.movedim(-2, -3)
        estimates[m] = float(rs._diagonal_gemm_expansion_error(aq, aq, g_k, m).max())
        assert estimates[m] >= float((cost - reference).abs().max()), (m, estimates[m])
    assert errors[8] <= errors[0], errors
    assert estimates[8] < estimates[0], estimates


def test_rotation_aware_shift_changes_the_live_route_at_rounding_level_only() -> None:
    base = _model_step(_live_config())
    on = _model_step(_live_config(pair_kl_gemm_shift="rotation_aware"))
    assert float((on[0] - base[0]).abs()) < 1e-5 and _max_relative(on[1], base[1]) < 1e-4


def test_pair_kl_gemm_shift_field_is_validated_and_marked_off_its_route() -> None:
    assert VFE3Config.__dataclass_fields__["pair_kl_gemm_shift"].default == "first_key"
    with pytest.raises(ValueError, match="pair_kl_gemm_shift"):
        _live_config(pair_kl_gemm_shift="zero")
    assert "pair_kl_gemm_shift" not in _live_config().inactive_fields()
    for off in (dict(relational_routing="noisy_query"), dict(rope_mode="none"),
                dict(projected_factorization="gram_certified")):
        assert "pair_kl_gemm_shift" in _live_config(**off).inactive_fields(), off


# ------------------------------------------------------------------------------------------------ config

def test_new_fields_default_off_and_follow_the_route_in_inactive_fields() -> None:
    for name in ("relational_type_moments", "relational_shared_canonical_mean"):
        assert VFE3Config.__dataclass_fields__[name].default is False
        assert name not in _live_config().inactive_fields()
        assert name in _live_config(rope_mode="none").inactive_fields()
        with pytest.raises(ValueError, match=name):
            _live_config(**{name: 1})
    assert "relational_type_moments" in _live_config(pos_phi="learned").inactive_fields()
    assert "relational_type_moments" in _live_config(phi_type_dropout=0.1).inactive_fields()
    assert "relational_shared_canonical_mean" not in _live_config(phi_type_dropout=0.1).inactive_fields()


# ------------------------------------------------------------------------------------------------ A012

def _conditioned(kappa: float, rows: int, cols: int, gen: torch.Generator) -> torch.Tensor:
    left = torch.linalg.qr(torch.randn(rows, rows, generator=gen, dtype=F64))[0]
    right = torch.linalg.qr(torch.randn(cols, rows, generator=gen, dtype=F64))[0]
    return (left * torch.logspace(0, -math.log10(kappa), rows, dtype=F64)) @ right.T


def _aligned_case(kappa: float, seed: int = 0) -> Tuple[List[torch.Tensor], torch.Tensor]:
    r"""Aligned ill-conditioned comparison maps (S_k = S_q R, R a small rotation): every pair is attention-relevant
    and the key precision is as ill-conditioned as the query covariance (the case of the audit's probe
    gap-certified-factorization-routes_qr2_aligned_fd.py, part (a))."""
    g = torch.Generator().manual_seed(seed)
    N, H, r, K = 6, 1, 8, 8
    S_q = _conditioned(kappa, r, K, g)[None]
    A = 0.02 * torch.randn(K, K, generator=g, dtype=F64)
    S_k = S_q @ torch.linalg.matrix_exp(A - A.T)
    mu = 0.3 * torch.randn(1, N, K, generator=g, dtype=F64)
    sigma = torch.exp(0.05 * torch.randn(1, N, K, generator=g, dtype=F64))
    inverse = torch.linalg.matrix_exp(0.02 * torch.randn(1, N, H, K, K, generator=g, dtype=F64))
    return [mu, sigma, inverse, S_q, S_k], torch.randn(1, H, N, N, generator=g, dtype=F64)


def _aligned_score(mu: torch.Tensor, sigma: torch.Tensor, inverse: torch.Tensor, S_q: torch.Tensor,
                   S_k: torch.Tensor) -> torch.Tensor:
    N, K = mu.shape[-2:]
    mu_h, sig_h = mu.unsqueeze(-2).expand(1, N, 1, K), sigma.unsqueeze(-2).expand(1, N, 1, K)
    aq, Gq = rs.projected_moments(mu_h, sig_h, inverse, S_q)
    ak, Gk = rs.projected_moments(mu_h, sig_h, inverse, S_k)
    return rs.projected_gaussian_kl(aq, Gq, ak, Gk, projection_q=(sig_h, inverse, S_q, mu_h),
                                    projection_k=(sig_h, inverse, S_k, mu_h), symmetric_inputs=True)


def _aligned_run(inputs: List[torch.Tensor], weights: torch.Tensor, dtype: torch.dtype
                 ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    leaves = [x.detach().to(dtype).clone().requires_grad_(True) for x in inputs]
    D = _aligned_score(*leaves)
    (D * weights.to(dtype)).sum().backward()
    return D.detach().double(), [x.grad.double() for x in leaves]


def test_certified_factor_route_gradients_keep_the_factor_accuracy() -> None:
    r"""At kappa(S) = 300 every entry passes the factor certificate and fails the Gram certificate. Under
    ``cholesky_qr2_certified`` the float32 gradients stay within 1e-3 of float64 (the audit measured 1.8e-2 to 3.1e-2
    through the explicit-Gram GEMM), and no entry is counted as a rebuild."""
    inputs, weights = _aligned_case(300.0)
    rs.set_projected_condition_recovery(False)
    rs.set_projected_factor_route("gram")
    D64, g64 = _aligned_run(inputs, weights, F64)
    rs.set_projected_condition_recovery(True)
    rs.set_projected_factor_route("cholesky_qr2")
    rs.reset_source_recovery_elements()
    D32, g32 = _aligned_run(inputs, weights, F32)
    assert rs.source_recovery_elements() == 0
    assert float((D32 - D64).abs().max()) < 1e-4
    for name, a, b in zip(("mu", "sigma", "inverse", "S_q", "S_k"), g32, g64):
        assert float((a - b).norm() / b.norm()) < 1e-3, name


# ------------------------------------------------------------------------------------------------ A013

def _noisy_key_case(kappa_k: float, noise: float, seed: int = 0) -> Tuple[torch.Tensor, ...]:
    r"""A well-conditioned query map and a key map of condition ``kappa_k``, whose broadened key law
    G_k + n I fails the float32 Gram certificate for a small noise n."""
    g = torch.Generator().manual_seed(seed)
    N, H, r, K = 8, 1, 4, 4
    S_q, S_k = _conditioned(3.0, r, K, g)[None], _conditioned(kappa_k, r, K, g)[None]
    mu = 3.0 * torch.randn(1, N, K, generator=g, dtype=F64)
    mu[0, 0] += 6.0
    sigma = torch.exp(0.1 * torch.randn(1, N, K, generator=g, dtype=F64))
    inverse = torch.eye(K, dtype=F64).expand(1, N, H, K, K).clone()
    return mu, sigma, inverse, S_q, S_k, torch.tensor([noise], dtype=F64)


def _noisy_key_score(mu: torch.Tensor, sigma: torch.Tensor, inverse: torch.Tensor, S_q: torch.Tensor,
                     S_k: torch.Tensor, noise: torch.Tensor, **options: Any) -> torch.Tensor:
    N, K = mu.shape[-2:]
    mu_h, sig_h = mu.unsqueeze(-2).expand(1, N, 1, K), sigma.unsqueeze(-2).expand(1, N, 1, K)
    aq, Gq = rs.projected_moments(mu_h, sig_h, inverse, S_q)
    ak, Gk = rs.projected_moments(mu_h, sig_h, inverse, S_k)
    return rs.projected_noisy_key_cost(aq, Gq, ak, Gk, key_noise=noise, projection_q=(sig_h, inverse, S_q, mu_h),
                                       projection_k=(sig_h, inverse, S_k, mu_h), symmetric_inputs=True, **options)


def test_source_root_of_the_broadened_key_law_is_the_augmented_root() -> None:
    mu, sigma, inverse, S_q, S_k, noise = _noisy_key_case(30.0, 0.25)
    N, K = mu.shape[-2:]
    sig_h, mu_h = sigma.unsqueeze(-2).expand(1, N, 1, K), mu.unsqueeze(-2).expand(1, N, 1, K)
    _, Gk = rs.projected_moments(mu_h, sig_h, inverse, S_k)
    every = torch.ones(1, N, 1, dtype=torch.bool)
    root = rs._source_root(Gk, (sig_h, inverse, S_k, mu_h, noise), every)                     # (N, r, K + r)
    assert root.shape == (N, 4, K + 4)
    broadened = Gk + noise[:, None, None] * torch.eye(4, dtype=F64)
    assert torch.allclose(root @ root.transpose(-1, -2), broadened.reshape(N, 4, 4), rtol=0, atol=1e-12)
    plain = rs._source_root(Gk, (sig_h, inverse, S_k, mu_h), every)                           # (N, r, K)
    assert torch.equal(root[..., :K], plain)


@pytest.mark.parametrize("route", ["gram", "cholesky_qr2"])
def test_certified_noisy_key_rebuilds_flagged_keys_from_the_broadened_law(route: str) -> None:
    inputs = _noisy_key_case(1e3, 1e-4)
    rs.set_projected_condition_recovery(False)
    rs.set_projected_factor_route("gram")
    D64 = _noisy_key_score(*inputs).double()
    rs.set_projected_condition_recovery(True)
    rs.set_projected_factor_route(route)            # the broadened law has no factor route: both act as gram_certified
    rs.reset_source_recovery_elements()
    D32 = _noisy_key_score(*(x.float() for x in inputs)).double()
    relevant = D64 - D64.amin(-1, keepdim=True) < 30.0
    assert rs.source_recovery_elements() >= 8                                   # every key entry is rebuilt
    # The committed route kept the flagged key factors: 21.6 nats (1.5e-4 relative) on these pairs; rebuilt, 0.15
    # nats (8.4e-7 relative), at costs up to 1.8e5 nats.
    error = (D32 - D64).abs()[relevant]
    assert float(error.max()) < 1.0
    assert float((error / D64.abs()[relevant].clamp_min(1.0)).max()) < 1e-5


def test_gemm_fallback_estimate_excludes_only_rebuilt_keys() -> None:
    r"""Without a key source tuple a flagged key keeps its factor, so it stays in the GEMM error estimate and the
    fallback fires as it does without the certificate."""
    mu, sigma, inverse, S_q, S_k, noise = (x.float() for x in _noisy_key_case(1e3, 1e-4))
    N, K = mu.shape[-2:]
    mu_h, sig_h = mu.unsqueeze(-2).expand(1, N, 1, K), sigma.unsqueeze(-2).expand(1, N, 1, K)
    aq, Gq = rs.projected_moments(mu_h, sig_h, inverse, S_q)
    ak, Gk = rs.projected_moments(mu_h, sig_h, inverse, S_k)
    G_noisy = Gk + noise[:, None, None] * torch.eye(4)
    counts = {}
    for recovery in (False, True):
        rs.set_projected_condition_recovery(recovery)
        rs.reset_gemm_fallback_slices()
        rs.projected_gaussian_kl(aq, Gq, ak, G_noisy, projection_q=(sig_h, inverse, S_q, mu_h), projection_k=None,
                                 symmetric_inputs=True, gemm_expansion_tol=1e-3)
        counts[recovery] = rs.gemm_fallback_slices()
    assert counts == {False: 1, True: 1}


# ------------------------------------------------------------------------------------------------ A179

def _recovery_case(small: float = 1e-4, dtype: torch.dtype = F32, tokens: int = 8,
                   ill: int = 3) -> Tuple[torch.Tensor, ...]:
    r"""Two sequences of ``tokens`` tokens, two heads, rank 3. Token ``ill`` of sequence 0 has four of its six variances
    at ``small``, so its comparison covariances are ill-conditioned: at 1e-4 they factor in float32 and fail the Gram
    certificate."""
    g = torch.Generator().manual_seed(11)
    B, N, H, r, K = 2, tokens, 2, 3, 6
    mu = torch.randn(B, N, K, generator=g, dtype=F64)
    sigma = 0.5 + torch.rand(B, N, K, generator=g, dtype=F64)
    sigma[0, ill, :4] = small
    inverse = torch.eye(K // H, dtype=F64) + 0.1 * torch.randn(B, N, H, K // H, K // H, generator=g, dtype=F64)
    S_q = torch.randn(H, r, K, generator=g, dtype=F64)
    S_k = torch.randn(H, r, K, generator=g, dtype=F64)
    return tuple(x.to(dtype) for x in (mu, sigma, inverse, S_q, S_k))


def _recovery_score(mu: torch.Tensor, sigma: torch.Tensor, inverse: torch.Tensor, S_q: torch.Tensor,
                    S_k: torch.Tensor, cost: Any = None) -> torch.Tensor:
    B, N, K = mu.shape
    H = S_q.shape[0]
    mu_h, sig_h = mu.unsqueeze(-2).expand(B, N, H, K), sigma.unsqueeze(-2).expand(B, N, H, K)
    aq, Gq = rs.projected_moments(mu_h, sig_h, inverse, S_q)
    ak, Gk = rs.projected_moments(mu_h, sig_h, inverse, S_k)
    return (cost or rs.projected_gaussian_kl)(aq, Gq, ak, Gk, projection_q=(sig_h, inverse, S_q, mu_h),
                                              projection_k=(sig_h, inverse, S_k, mu_h), symmetric_inputs=True)


def _relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(((a.double() - b.double()).abs() / b.double().abs().clamp_min(1.0)).max())


def test_recovery_scores_only_the_rows_and_columns_of_recovered_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = _recovery_case()
    rs.set_projected_condition_recovery(False)
    D64 = _recovery_score(*(x.double() for x in inputs))
    pairs: List[int] = []
    statuses: List[int] = []
    real_tile, real_status = rs._projected_kl_tile, rs._cholesky_status

    def tile(aq, ak, q_matrix, k_matrix, logdet_q, logdet_k, factor_trace):  # type: ignore[no-untyped-def]
        if factor_trace:
            pairs.append(aq.shape[0] * aq.shape[-2] * ak.shape[-2])
        return real_tile(aq, ak, q_matrix, k_matrix, logdet_q, logdet_k, factor_trace)

    def status(*args, **kwargs):  # type: ignore[no-untyped-def]
        statuses.append(1)
        return real_status(*args, **kwargs)

    monkeypatch.setattr(rs, "_projected_kl_tile", tile)
    monkeypatch.setattr(rs, "_cholesky_status", status)
    rs.set_projected_condition_recovery(True)
    rs.reset_source_recovery_elements()
    with torch.no_grad():
        D32 = _recovery_score(*inputs)
    assert rs.source_recovery_elements() >= 2                   # the ill token's entries, query and key roles
    # One recovered token in sequence 0: its query row (8 keys) and its key column (7 more queries) in each slice that
    # holds it, at most 2 * (8 + 7) pairs; the former route scored every pair of those slices, 2 * 8 * 8.
    assert 0 < sum(pairs) <= 2 * (8 + 7)
    assert len(statuses) == 2                                   # one status per role; no role is refactored
    assert _relative_error(D32, D64) < 1e-3


@pytest.mark.parametrize("tokens,ill,budget", [(8, 3, None), (40, 35, 18432)])
def test_recovered_pairs_take_the_factor_tile_value_and_the_rest_keep_the_gemm(
        tokens: int, ill: int, budget: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    r"""Every pair touching a recovered entry carries the factor-tile value of its two entries, and every other pair
    the value of the call without recovery, bit for bit. The committed token flags read the head axis of G
    (``shape[-3]``) instead of the token axis: with more than 32 tokens and a pair budget of 32 x 8 pairs (the second
    case) its tile loop skipped the recovered row of token 35 beyond the first column tile."""
    if budget is not None:
        monkeypatch.setattr(rs, "_RECOVERY_PAIR_TILE_BYTES", budget)
    inputs = _recovery_case(tokens=tokens, ill=ill)
    mu, sigma, inverse, S_q, S_k = inputs
    B, N, K = mu.shape
    H = S_q.shape[0]
    rs.set_projected_condition_recovery(False)
    with torch.no_grad():
        D_plain = _recovery_score(*inputs)
    captured: List[Tuple[torch.Tensor, Any]] = []
    real = rs._strict_cholesky

    def spy(G, projection=None, prepared=None, factored=False, commutant_pairs=None, **kwargs):  # type: ignore[no-untyped-def]
        out = real(G, projection, prepared, factored, commutant_pairs, **kwargs)
        if prepared is not None:
            captured.append(out)
        return out

    monkeypatch.setattr(rs, "_strict_cholesky", spy)
    rs.set_projected_condition_recovery(True)
    with torch.no_grad():
        D = _recovery_score(*inputs)
        (chol_q, rec_q), (chol_k, rec_k) = captured
        assert rec_q is not None and rec_k is not None
        mu_h, sig_h = mu.unsqueeze(-2).expand(B, N, H, K), sigma.unsqueeze(-2).expand(B, N, H, K)
        aq = rs.projected_moments(mu_h, sig_h, inverse, S_q)[0].movedim(-2, -3)              # (B, H, N, r)
        ak = rs.projected_moments(mu_h, sig_h, inverse, S_k)[0].movedim(-2, -3)
        logdet_q, logdet_k = ((2 * c.diagonal(dim1=-2, dim2=-1).log().sum(-1)).transpose(-1, -2)
                              for c in (chol_q, chol_k))                                     # (B, H, N)
        full = rs._projected_kl_tile(aq, ak, chol_q.movedim(-3, -4), chol_k.movedim(-3, -4), logdet_q, logdet_k,
                                     True)                                                    # (B, H, N, N)
    touch = rec_q.movedim(-1, -2).unsqueeze(-1) | rec_k.movedim(-1, -2).unsqueeze(-2)       # (B, H, N, N)
    assert bool(touch[0, :, ill, :].all()) and bool(touch[0, :, :, ill].all()) and not bool(touch[1].any())
    assert int(touch.sum()) == 2 * (2 * tokens - 1)                         # the ill token's row and column, two heads
    assert torch.equal(D, torch.where(touch, full, D_plain))


def test_a_failed_entry_overwrites_a_safe_factor_without_refactoring(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""A lost-SPD entry is rebuilt from its source root into the prepared :class:`_GramPrecision` factor, which masks
    a failed entry's gradient, without the refactorization a plain factor needs."""
    g = torch.Generator().manual_seed(2)
    n, r, K = 5, 3, 4
    S = torch.randn(1, r, K, generator=g, dtype=F64)                                       # (H, r, K)
    sigma = 0.5 + torch.rand(1, n, 1, K, generator=g, dtype=F64)                            # (1, n, H, K)
    inverse = torch.eye(K, dtype=F64).expand(1, n, 1, K, K)                                 # (1, n, H, K, K)
    X = S * sigma.sqrt().unsqueeze(-2)                                                      # (1, n, H, r, K)
    G0 = (X @ X.transpose(-1, -2)).float()
    G0[0, 2, 0] = torch.diag(torch.tensor([1.0, 1.0, -1e-3]))                              # rounding lost SPD
    projection = (sigma.float(), inverse.float(), S.float())
    calls: List[int] = []
    real = rs._cholesky_status

    def status(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(1)
        return real(*args, **kwargs)

    factors = {}
    for safe in (True, False):
        G = G0.clone().requires_grad_(True)
        chol, failed, P, _ = rs._GramPrecision.apply(G)
        prepared = rs._cholesky_status(G, True, factor=(chol, failed, P))
        calls.clear()
        monkeypatch.setattr(rs, "_cholesky_status", status)
        out, rebuilt = rs._strict_cholesky(G, projection, (*prepared[:3], prepared[3].tolist()), False, None,
                                           safe_factor=safe)
        monkeypatch.setattr(rs, "_cholesky_status", real)
        assert len(calls) == (0 if safe else 1)
        assert rebuilt is not None and rebuilt.nonzero().tolist() == [[0, 2, 0]]
        grad, = torch.autograd.grad(out.square().sum(), G)
        assert bool(torch.isfinite(grad).all()) and float(grad[0, 2, 0].abs().max()) == 0.0
        factors[safe] = out.detach()
    assert torch.equal(factors[True], factors[False])


def test_noisy_query_recovery_rescores_only_the_recovered_slices(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = _recovery_case()
    rs.set_projected_condition_recovery(False)
    D64 = _recovery_score(*(x.double() for x in inputs), cost=rs.projected_noisy_query_cost)
    shapes: List[Tuple[int, int, int]] = []
    real_tile = rs._noisy_query_tile

    def tile(aq, ak, Gq, Gk, logdet_q):  # type: ignore[no-untyped-def]
        if aq.dtype == F64:
            shapes.append((aq.shape[0], aq.shape[-2], ak.shape[-2]))
        return real_tile(aq, ak, Gq, Gk, logdet_q)

    monkeypatch.setattr(rs, "_noisy_query_tile", tile)
    rs.set_projected_condition_recovery(True)
    with torch.no_grad():
        D32 = _recovery_score(*inputs, cost=rs.projected_noisy_query_cost)
    # Only slices of sequence 0 (two heads) are rescored, and in them only the recovered token's row and column; the
    # former route rescored all 2 * 2 slices, 4 * 8 * 8 pairs.
    assert shapes and all(n <= 2 for n, _, _ in shapes)
    assert sum(n * i * j for n, i, j in shapes) <= 2 * (8 + 7)
    assert _relative_error(D32, D64) < 1e-3


# ------------------------------------------------------------------------------------------------ A311

def test_tile_loop_record_reads_zero_on_the_untiled_route_and_counts_a_tiled_one() -> None:
    from vfe3.run_artifacts import _realized_diag_moments_kernel
    rs.reset_diagonal_moments_triton_calls()
    _model_step(_live_config())
    assert rs.projection_tile_loops() == 0
    assert _realized_diag_moments_kernel()["relational_tile_loops"] == 0
    rs.set_projection_tile_bytes(256)                         # a budget below one token's projected maps
    with torch.no_grad():
        _recovery_score(*_recovery_case())
    assert rs.projection_tile_loops() > 0
    assert _realized_diag_moments_kernel()["relational_tile_loops"] == rs.projection_tile_loops()
    rs.reset_diagonal_moments_triton_calls()
    assert rs.projection_tile_loops() == 0


# ------------------------------------------------------------------------------------------------ A353

def test_relational_mixture_mean_is_a_labelled_test_oracle_with_no_package_caller() -> None:
    assert "Test oracle" in (rs.relational_mixture_mean.__doc__ or "")
    callers = [path for path in (REPO / "vfe3").rglob("*.py")
               if path.name != "relational_scoring.py" and "relational_mixture_mean" in path.read_text(encoding="utf-8")]
    assert callers == []
