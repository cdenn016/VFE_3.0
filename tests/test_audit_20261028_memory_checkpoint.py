r"""Remediation of the 2026-10-28 audit: F01 and F02 (docs/audit-results_2026-10-28.md).

F01. The low-rank diagonal variance formed :math:`\sum_{l\ne k}p_l(U_k\cdot V_l)^2` as :math:`[UGU^\top]_{kk} - p_ka_k^2`
with :math:`G = V^\top PV`. Where the self term :math:`p_kV_kV_k^\top` dominates :math:`G`, accumulating :math:`G`
rounds the off-axis remainder away: for :math:`F = [[0, 10^{-4}], [0, 1]]` at record and cue variance 4096 the float32
result variance was the result noise :math:`e^{-25}` instead of :math:`2.048\cdot10^{-5}`, with a zero
:math:`\partial/\partial U_{000}`. Both routes now contract :math:`U_k` with the leave-one-out Gram
:math:`G^{(-k)} = \sum_{l\ne k}p_lV_lV_l^\top`, an exclusive prefix plus an exclusive suffix over :math:`l`, which
contains no self term and no subtraction.

F02. A failed periodic checkpoint on the last step suppressed ``checkpoint_at_end``, which skipped whenever the final
step was a multiple of the interval. The end save now skips only when that final step was saved successfully.

Tiny CPU cases only (K <= 6, M <= 9); the interpreter check runs in a subprocess as in
``tests/test_cue_result_triton_20261025.py``.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch
from torch.utils.data import DataLoader

import vfe3.run_artifacts as ra
import vfe3.train as T
from tests.profiles import test_config
from tests.test_cue_result_triton_20261025 import _NO_TRITON, _run_script, _subprocess_env
from vfe3.data.datasets import TokenWindows
from vfe3.model.cue_result_memory import cue_result_diagonal_posterior
from vfe3.model.model import VFEModel

CPU = torch.device("cpu")


def _fixture(dtype: torch.dtype, off_axis: float = 1e-4) -> Dict[str, torch.Tensor]:
    r"""K = 2, rank 2, one component: :math:`F = I + UV^\top = [[0, \epsilon], [0, 1]]`, zero means, record and cue
    variances 4096, result noise :math:`e^{-25}`."""
    return {"xi": torch.zeros(1, 2, dtype=dtype), "c": torch.full((1, 2), 4096.0, dtype=dtype),
            "nu": torch.zeros(1, 2, dtype=dtype), "omega": torch.zeros(1, 2, dtype=dtype),
            "log_lambda": torch.full((1, 2), math.log(4096.0), dtype=dtype),
            "log_result_noise": torch.full((1, 2), -25.0, dtype=dtype), "log_pi": torch.zeros(1, dtype=dtype),
            "regression_left": torch.tensor([[[1.0, 0.0], [0.0, 0.0]]], dtype=dtype),
            "regression_right": torch.tensor([[[-1.0, 0.0], [off_axis, 0.0]]], dtype=dtype)}


def _posterior(ops: Dict[str, torch.Tensor], **kwargs: Any):
    return cue_result_diagonal_posterior(ops["xi"], ops["c"], ops["nu"], ops["omega"], ops["log_lambda"],
                                         ops["log_result_noise"], ops["log_pi"],
                                         regression_left=ops["regression_left"],
                                         regression_right=ops["regression_right"], **kwargs)


def test_f01_float32_keeps_the_off_axis_variance_and_the_factor_gradient() -> None:
    ops = _fixture(torch.float32)
    ops["regression_left"].requires_grad_(True)
    var0 = _posterior(ops).covariance[0, 0]
    var0.backward()
    p = 4096.0 * 4096.0 / 8192.0                                              # diag P_X = lambda c / (lambda + c)
    exact = p * 1e-8 + math.exp(-25.0)                                        # F_01^2 p + q
    assert var0.item() == pytest.approx(exact, rel=1e-5)
    # d var_0 / d U_000 = 2 p (1 + a_0) V_00 + 2 sum_{l != 0} p_l (U_0 . V_l) V_l0 = 0 + 2 p * 1e-8
    assert ops["regression_left"].grad[0, 0, 0].item() == pytest.approx(2.0 * p * 1e-8, rel=1e-4)


def _dense_reference(ops: Dict[str, torch.Tensor]) -> torch.Tensor:
    r"""The float64 dense route on F = I + U V^T: the independent sum :math:`\sum_lF_{kl}^2p_l`."""
    d = {k: v.detach().double() for k, v in ops.items()}
    F = torch.eye(d["nu"].shape[-1], dtype=torch.float64) + d["regression_left"] @ d["regression_right"].transpose(-1, -2)
    return cue_result_diagonal_posterior(d["xi"], d["c"], d["nu"], d["omega"], d["log_lambda"],
                                         d["log_result_noise"], d["log_pi"], regression_offset=F - torch.eye(F.shape[-1],
                                         dtype=torch.float64)).covariance


def test_f01_near_cancelling_rows_match_the_dense_float64_variance() -> None:
    g = torch.Generator().manual_seed(4)
    R, M, K, rank = 3, 1, 6, 2                  # one component at zero means: no between-component spread
    U = 0.5 * torch.randn(M, K, rank, generator=g, dtype=torch.float64)
    V = 1e-4 * torch.randn(M, K, rank, generator=g, dtype=torch.float64)      # small off-axis couplings
    for a in range(M):                                                        # row k = 0 cancels its own coordinate
        V[a, 0] = -U[a, 0] / U[a, 0].square().sum()
    ops = {"xi": torch.zeros(R, K, dtype=torch.float64),
           "c": 1000.0 + torch.rand(R, K, generator=g, dtype=torch.float64),
           "nu": torch.zeros(M, K, dtype=torch.float64),
           "omega": torch.zeros(M, K, dtype=torch.float64),
           "log_lambda": 7.0 + 0.1 * torch.randn(M, K, generator=g, dtype=torch.float64),
           "log_result_noise": torch.full((M, K), -30.0, dtype=torch.float64),
           "log_pi": torch.randn(M, generator=g, dtype=torch.float64),
           "regression_left": U, "regression_right": V}
    ref = _dense_reference(ops)
    for dtype, tol in ((torch.float64, 1e-10), (torch.float32, 1e-3)):
        got = _posterior({k: v.to(dtype) for k, v in ops.items()}).covariance.double()
        assert ((got - ref).abs() / ref).max().item() < tol, dtype


_F01_INTERPRETER = """
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
from tests.test_audit_20261028_memory_checkpoint import _fixture, _posterior
assert tk.INTERPRETED
fused_kernels.fused_route_active = lambda xi: True
ops = _fixture(torch.float32)
ops["regression_left"].requires_grad_(True)
var0 = _posterior(ops, fused=True).covariance[0, 0]
var0.backward()
print(json.dumps({"var0": var0.item(), "dU": ops["regression_left"].grad[0, 0, 0].item()}))
"""


def test_f01_interpreted_triton_keeps_the_off_axis_variance_and_the_factor_gradient() -> None:
    report = _run_script(_F01_INTERPRETER, _subprocess_env(TRITON_INTERPRET="1"))
    if report == _NO_TRITON:
        pytest.skip("triton unavailable")
    p = 2048.0
    assert report["var0"] == pytest.approx(p * 1e-8 + math.exp(-25.0), rel=1e-5)
    assert report["dU"] == pytest.approx(2.0 * p * 1e-8, rel=1e-4)


# ------------------------------------------------------------------------------------------------ F02

V_SIZE, B, N = 11, 2, 4


@pytest.fixture()
def _quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(T, "_save_eval_attention_maps", lambda *args, **kwargs: None)
    monkeypatch.setattr(ra, "_run_figures_isolated", lambda *args, **kwargs: True)
    torch.set_num_threads(1)


def _train(run_dir: Path, fail_at: set, monkeypatch: pytest.MonkeyPatch) -> List[int]:
    cfg = test_config(vocab_size=V_SIZE, embed_dim=4, n_heads=2, batch_size=B, max_seq_len=N, eval_stride=None,
                      n_layers=1, n_e_steps=1, warmup_steps=0, max_steps=4, log_interval=0, eval_interval=0,
                      checkpoint_interval=2, checkpoint_at_end=True, skip_belief_sigma_update=True,
                      generate_embedding_figures=False, use_block_mlp=False, family="gaussian_diagonal",
                      gauge_group="block_glk", attention_mode="projected_predictive", attention_residual=True,
                      decode_mode="linear")
    torch.manual_seed(5)
    model = VFEModel(cfg)
    artifacts = ra.RunArtifacts(run_dir, cfg, model)
    calls: List[int] = []
    original = artifacts.save_checkpoint

    def save(step: int, *args: Any, **kwargs: Any) -> Any:
        calls.append(step)
        if step in fail_at:
            fail_at.discard(step)                                             # a one-time transient failure
            raise OSError("injected transient write failure")
        return original(step, *args, **kwargs)

    monkeypatch.setattr(artifacts, "save_checkpoint", save)
    train = TokenWindows((torch.arange(41) * 5 + 2) % V_SIZE, N)
    T.train(model, DataLoader(train, batch_size=B, shuffle=False, drop_last=True), cfg, n_steps=4,
            grad_clip=cfg.grad_clip, log_interval=0, eval_interval=0, val_loader=None, device=CPU,
            logger=logging.getLogger("audit-20261028"), artifacts=artifacts, generate_samples=False)
    return calls


def test_f02_failed_final_periodic_checkpoint_is_retried_at_the_end(tmp_path: Path, _quiet: None,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _train(tmp_path / "failed", {4}, monkeypatch)
    assert calls == [2, 4, 4]
    assert (tmp_path / "failed" / "checkpoints" / "step_4.pt").exists()


def test_f02_successful_final_periodic_checkpoint_is_not_duplicated(tmp_path: Path, _quiet: None,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    assert _train(tmp_path / "clean", set(), monkeypatch) == [2, 4]
