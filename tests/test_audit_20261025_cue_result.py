r"""Cue/result memory remediation of the 2026-10-25 audit: F01, F02 and F15 (docs/audit-results_2026-10-25.md).

F01. At large finite log densities the absolute log evidence lost its :math:`\log l` increment in float32, so two
identical components each carried weight one (mass two, mean 10,000 and covariance 50,000,004 where the answer is
weights one half, mean 5,000 and variance 1.5), and the tile budget changed the covariance by seven orders of
magnitude. The tiles now carry a row shift and a scaled mass, and the log weights are formed relative to the shift.

F02. The low-rank diagonal variance :math:`q + p + 2pa + [UGU^\top]_{kk}` cancelled a positive result noise where
:math:`F = I + UV^\top` cancels a coordinate (``F = 0``, ``Q = e^{-25}`` returned 0). It is now
:math:`p(1 + a)^2 + \max([UGU^\top]_{kk} - pa^2, 0) + q`, never below :math:`q`.

F15. The dense kernel tiled only components, so one component across all tokens could exceed the budget, and no tile
was checkpointed. It now tiles tokens and components and checkpoints each tile when gradients are on.

Each case runs the torch tiles; the fused route's F01 and F02 cases run under the Triton interpreter in a subprocess.
CPU only, one thread, tiny shapes.
"""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch

import vfe3.model.cue_result_memory as crm
from vfe3.model.cue_result_memory import cue_result_diagonal_posterior, cue_result_exact_posterior

torch.set_num_threads(1)
REPO = Path(__file__).resolve().parents[1]
F32 = torch.float32


def _large_logmass(M: int, dtype: torch.dtype = F32) -> Dict[str, torch.Tensor]:
    r"""The audit fixture: K = 1, xi = 10,000, C = 1, M identical components nu = omega = 0, lambda = Q = 1, F = I."""
    z = torch.zeros(M, 1, dtype=dtype)
    return {"xi": torch.full((1, 1), 1e4, dtype=dtype), "c": torch.ones(1, 1, dtype=dtype), "nu": z, "omega": z,
            "log_lambda": z, "log_result_noise": z, "log_pi": torch.zeros(M, dtype=dtype),
            "regression_left": torch.zeros(M, 1, 1, dtype=dtype), "regression_right": torch.zeros(M, 1, 1, dtype=dtype)}


def _diag(ops: Dict[str, torch.Tensor], tile: int) -> crm.CueResultPosterior:
    return cue_result_diagonal_posterior(ops["xi"], ops["c"], ops["nu"], ops["omega"], ops["log_lambda"],
                                         ops["log_result_noise"], ops["log_pi"],
                                         regression_left=ops["regression_left"],
                                         regression_right=ops["regression_right"], tile_bytes=tile)


def _dense(ops: Dict[str, torch.Tensor], tile: int) -> crm.CueResultPosterior:
    M, K = ops["nu"].shape
    offset = torch.einsum("akr,alr->akl", ops["regression_left"], ops["regression_right"])
    return cue_result_exact_posterior(ops["xi"], torch.diag_embed(ops["c"]), ops["nu"], ops["omega"],
                                      ops["log_lambda"], ops["log_result_noise"], ops["log_pi"],
                                      regression_offset=offset, tile_bytes=tile)


# ------------------------------------------------------------------------------------------------ F01

@pytest.mark.parametrize("kernel", ["diagonal", "dense"])
@pytest.mark.parametrize("tile", [1, 200, 10 ** 9])                  # one-component tiles, unequal tiles, one tile
def test_f01_identical_components_at_a_large_log_density_offset(kernel: str, tile: int) -> None:
    ops = {k: v.clone().requires_grad_(v.is_floating_point()) for k, v in _large_logmass(3).items()}
    out = (_diag if kernel == "diagonal" else _dense)(ops, tile)
    var = out.covariance.reshape(-1)
    assert float(out.log_evidence) < -2e7                               # the regime of the failure (-2.5e7)
    assert torch.allclose(out.log_weights.exp(), torch.full((1, 3), 1 / 3), rtol=1e-6, atol=0)
    assert float(out.mean) == pytest.approx(5000.0, rel=1e-7)
    assert float(var) == pytest.approx(1.5, rel=1e-6)
    loss = out.mean.sum() + out.covariance.sum() + (out.log_weights.exp() * torch.arange(3.0)).sum()
    grads = torch.autograd.grad(loss, [t for t in ops.values() if t.requires_grad], allow_unused=True)
    assert all(g is None or torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("kernel", ["diagonal", "dense"])
@pytest.mark.parametrize("tile", [1, 200, 10 ** 9])
def test_f01_distinct_results_merge_correctly_at_a_large_log_density_offset(kernel: str, tile: int) -> None:
    r"""Five components with the same cue (equal log densities near :math:`-2.5\cdot10^7`) and distinct result means
    and noises: the weights are 1/5 and the moments are those of the equal mixture, at every tile budget (one, a few
    and all components per tile). A common offset applied to the inputs themselves cannot be tested in float32: it
    rounds the inputs."""
    ops = _large_logmass(5)
    ops["omega"] = torch.tensor([[-2.0], [-0.5], [0.0], [1.0], [3.0]])
    ops["log_result_noise"] = torch.tensor([[-1.0], [0.0], [0.5], [-2.0], [0.25]])
    out = (_diag if kernel == "diagonal" else _dense)(ops, tile)
    omega, q = ops["omega"].double().squeeze(-1), ops["log_result_noise"].double().exp().squeeze(-1)
    mean = 5000.0 + omega.mean()
    var = 0.5 + q.mean() + omega.var(unbiased=False)
    assert torch.allclose(out.log_weights.exp(), torch.full((1, 5), 0.2), rtol=1e-6, atol=0)
    assert float(out.mean) == pytest.approx(float(mean), rel=1e-7)
    # The merged means sit near 5,000, where float32 spacing is 4.9e-4; the deviations about them carry that rounding.
    assert float(out.covariance.reshape(-1)) == pytest.approx(float(var), abs=2e-3)


# ------------------------------------------------------------------------------------------------ F02

def _canceled(K: int, rank: int, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    r"""F = I + U V^T cancels coordinate 0 (U_0 = e_0, V_0 = -e_0 on component 0), Q_0 = e^{-25}, c = lambda = 1."""
    left = torch.zeros(1, K, rank, dtype=dtype)
    right = torch.zeros(1, K, rank, dtype=dtype)
    left[0, 0, 0], right[0, 0, 0] = 1.0, -1.0
    if K > 1:                                                           # a live second coordinate and rank direction
        left[0, 1, rank - 1], right[0, 1, rank - 1] = 0.7, 0.4
    log_q = torch.zeros(1, K, dtype=dtype)
    log_q[0, 0] = -25.0
    return {"xi": torch.full((1, K), 0.3, dtype=dtype), "c": torch.ones(1, K, dtype=dtype),
            "nu": torch.zeros(1, K, dtype=dtype), "omega": torch.zeros(1, K, dtype=dtype),
            "log_lambda": torch.zeros(1, K, dtype=dtype), "log_result_noise": log_q,
            "log_pi": torch.zeros(1, dtype=dtype), "regression_left": left, "regression_right": right}


@pytest.mark.parametrize("K, rank", [(1, 1), (2, 2)])
def test_f02_canceled_coordinate_keeps_its_result_noise(K: int, rank: int) -> None:
    ops = _canceled(K, rank, F32)
    out = _diag(ops, 10 ** 9)
    dense64 = _dense({k: v.double() for k, v in ops.items()}, 10 ** 9)
    expected = dense64.covariance.diagonal(dim1=-2, dim2=-1)
    assert float(expected[0, 0]) == pytest.approx(1.3887943746404563e-11, rel=1e-6)
    assert float(out.covariance[0, 0]) == pytest.approx(float(expected[0, 0]), rel=1e-5)
    assert torch.allclose(out.covariance.double(), expected, rtol=1e-6)
    assert bool((out.covariance >= ops["log_result_noise"].exp()).all())


def test_f02_near_cancellation_values_and_gradients_follow_the_dense_reference() -> None:
    r"""Random low-rank regressions with one coordinate nearly canceled, float64: the rewritten variance and its
    gradients agree with the explicit dense :math:`F = I + UV^\top`."""
    g = torch.Generator().manual_seed(4)
    M, K, rank = 4, 3, 2
    left = 0.5 * torch.randn(M, K, rank, generator=g, dtype=torch.float64)
    right = 0.5 * torch.randn(M, K, rank, generator=g, dtype=torch.float64)
    left[:, 0, 0] = 1.0
    right[:, 0, 0] = -1.0 - 1e-4 - (left[:, 0, 1] * right[:, 0, 1])     # a_0 = -1 - 1e-4: F_00 = -1e-4
    ops = {"xi": torch.randn(2, K, generator=g, dtype=torch.float64),
           "c": 0.5 + torch.rand(2, K, generator=g, dtype=torch.float64),
           "nu": torch.randn(M, K, generator=g, dtype=torch.float64),
           "omega": torch.randn(M, K, generator=g, dtype=torch.float64),
           "log_lambda": 0.3 * torch.randn(M, K, generator=g, dtype=torch.float64),
           "log_result_noise": torch.full((M, K), -12.0, dtype=torch.float64),
           "log_pi": torch.randn(M, generator=g, dtype=torch.float64),
           "regression_left": left, "regression_right": right}

    def run(fn: Any) -> List[torch.Tensor]:
        leaves = {k: v.clone().requires_grad_(True) for k, v in ops.items()}
        out = fn(leaves, 10 ** 9)
        var = out.covariance if fn is _diag else out.covariance.diagonal(dim1=-2, dim2=-1)
        loss = (var * torch.arange(1.0, K + 1, dtype=torch.float64)).sum() + out.mean.sum()
        return [var.detach(), *torch.autograd.grad(loss, list(leaves.values()))]

    for a, b in zip(run(_diag), run(_dense)):
        assert torch.allclose(a, b, rtol=1e-9, atol=1e-12)


# ------------------------------------------------------------------------------------------------ F15

def test_f15_dense_kernel_tiles_tokens_within_the_budget_and_checkpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    g = torch.Generator().manual_seed(7)
    T, M, K = 8, 3, 2
    A = torch.randn(T, K, K, generator=g, dtype=torch.float64)
    ops = {"xi": torch.randn(T, K, generator=g, dtype=torch.float64),
           "C": A @ A.transpose(-1, -2) / K + 0.3 * torch.eye(K, dtype=torch.float64),
           "nu": torch.randn(M, K, generator=g, dtype=torch.float64),
           "omega": torch.randn(M, K, generator=g, dtype=torch.float64),
           "log_lambda": 0.3 * torch.randn(M, K, generator=g, dtype=torch.float64),
           "log_result_noise": torch.full((M, K), -1.0, dtype=torch.float64),
           "log_pi": torch.randn(M, generator=g, dtype=torch.float64),
           "regression_offset": 0.4 * torch.randn(M, K, K, generator=g, dtype=torch.float64)}
    per_pair = 6 * K * K * 8                                            # the kernel's float64 estimate per pair
    shapes: List[tuple] = []
    checkpointed: List[int] = []
    real_moments, real_checkpoint = crm._chunk_moments, crm.checkpoint

    def spy(xi: torch.Tensor, C: torch.Tensor, nu: torch.Tensor, *rest: Any) -> Any:
        shapes.append((xi.shape[0], nu.shape[0]))
        return real_moments(xi, C, nu, *rest)

    def spy_checkpoint(fn: Any, *args: Any, **kwargs: Any) -> Any:
        checkpointed.append(1)
        return real_checkpoint(fn, *args, **kwargs)

    monkeypatch.setattr(crm, "_chunk_moments", spy)
    monkeypatch.setattr(crm, "checkpoint", spy_checkpoint)

    def run(tile: int) -> List[torch.Tensor]:
        shapes.clear()
        checkpointed.clear()
        leaves = {k: v.clone().requires_grad_(True) for k, v in ops.items()}
        out = cue_result_exact_posterior(leaves["xi"], leaves["C"], leaves["nu"], leaves["omega"],
                                         leaves["log_lambda"], leaves["log_result_noise"], leaves["log_pi"],
                                         regression_offset=leaves["regression_offset"], tile_bytes=tile)
        loss = out.mean.sum() + out.covariance.square().sum() + out.log_evidence.sum()
        return [out.mean.detach(), out.covariance.detach(), *torch.autograd.grad(loss, list(leaves.values()))]

    whole = run(10 ** 9)
    assert shapes == [(T, M)] and checkpointed == []
    for pairs in (1, 2, 5):                                             # budgets of one, two and five pairs
        part = run(pairs * per_pair)
        assert all(t * m <= pairs for t, m in shapes), (pairs, shapes)
        assert len(shapes) == 2 * len(checkpointed)                     # each tile: forward, then recompute
        forward = shapes[:len(checkpointed)]
        assert sum(t for t, m in forward) * 1 >= T and sum(t * m for t, m in forward) == T * M
        for a, b in zip(part, whole):
            assert torch.allclose(a, b, rtol=1e-11, atol=1e-12)
    single = dict(ops, nu=ops["nu"][:1], omega=ops["omega"][:1], log_lambda=ops["log_lambda"][:1],
                  log_result_noise=ops["log_result_noise"][:1], log_pi=ops["log_pi"][:1],
                  regression_offset=ops["regression_offset"][:1])
    rows = {}
    for pairs in (4, 2):                                                # M = 1: the budget sets the token rows
        shapes.clear()
        with torch.no_grad():
            cue_result_exact_posterior(single["xi"], single["C"], single["nu"], single["omega"], single["log_lambda"],
                                       single["log_result_noise"], single["log_pi"],
                                       regression_offset=single["regression_offset"], tile_bytes=pairs * per_pair)
        rows[pairs] = max(t for t, _ in shapes)
    assert rows == {4: 4, 2: 2}


# ------------------------------------------------------------------------------------------------ fused route

_FUSED_SCRIPT = """
import json, sys
try:
    import triton  # noqa: F401
except Exception as exc:
    print("triton import failed:", exc)
    sys.exit(77)
import torch
torch.set_num_threads(1)
from vfe3.kernels import cue_result as fused_kernels
from vfe3.model.cue_result_memory import cue_result_diagonal_posterior
from tests.test_audit_20261025_cue_result import _canceled, _large_logmass
fused_kernels.fused_route_active = lambda xi: True
out = {}
ops = _large_logmass(3)
r = cue_result_diagonal_posterior(ops["xi"], ops["c"], ops["nu"], ops["omega"], ops["log_lambda"],
                                  ops["log_result_noise"], ops["log_pi"], regression_left=ops["regression_left"],
                                  regression_right=ops["regression_right"], fused=True)
out["f01"] = {"weights": r.log_weights.exp().tolist(), "mean": float(r.mean), "var": float(r.covariance)}
ops = _canceled(2, 2, torch.float32)
r = cue_result_diagonal_posterior(ops["xi"], ops["c"], ops["nu"], ops["omega"], ops["log_lambda"],
                                  ops["log_result_noise"], ops["log_pi"], regression_left=ops["regression_left"],
                                  regression_right=ops["regression_right"], fused=True)
out["f02"] = r.covariance.tolist()
print(json.dumps(out))
"""


def test_fused_route_f01_and_f02_under_the_triton_interpreter() -> None:
    env = dict(os.environ)
    env.update(TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="-1", VFE3_TEST_DEVICE="cpu", PYTHONPATH=str(REPO),
               OMP_NUM_THREADS="1")
    proc = subprocess.run([sys.executable, "-c", textwrap.dedent(_FUSED_SCRIPT)], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=900)
    if proc.returncode == 77:
        pytest.skip("triton unavailable")
    assert proc.returncode == 0, proc.stderr[-3000:]
    report = json.loads(proc.stdout.strip().splitlines()[-1])
    assert report["f01"]["weights"][0] == pytest.approx([1 / 3] * 3, rel=1e-6)
    assert report["f01"]["mean"] == pytest.approx(5000.0, rel=1e-7)
    assert report["f01"]["var"] == pytest.approx(1.5, rel=1e-6)
    ref = _dense({k: v.double() for k, v in _canceled(2, 2, torch.float32).items()}, 10 ** 9)
    expected = ref.covariance.diagonal(dim1=-2, dim2=-1)[0].tolist()
    assert report["f02"][0] == pytest.approx(expected, rel=1e-5)
