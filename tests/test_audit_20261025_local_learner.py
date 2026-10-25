r"""Local-learner remediation of the 2026-10-25 ultradeep audit (findings F04, F08, F19). Tiny CPU fixtures.

F04: importing the stage-0 local-learner test module leaves the process default dtype unchanged. F08: the
click-to-run launcher reuses a completed result only under the same effective arm settings, data protocol and
source identity. F19: the ridge transition M-step returns the expected residual covariance
:math:`E[(z - Wx)(z - Wx)^\top]` at the fitted W, not the unregularized shortcut.
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from vfe3.local_learner.core import init_params
from vfe3.local_learner.learners import Stats, m_step, residual_covariance, train_backprop, train_local

ROOT = Path(__file__).resolve().parents[1]


def test_importing_local_learner_tests_keeps_default_dtype() -> None:
    r"""A fresh interpreter imports the module (as collection does) and reports the default dtype before and after."""
    code = ("import torch; before = torch.get_default_dtype(); import tests.test_local_learner_20261021; "
            "print(before, torch.get_default_dtype())")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "VFE3_TEST_DEVICE": "cpu", "OMP_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "KMP_DUPLICATE_LIB_OK": "TRUE"}
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["torch.float32", "torch.float32"]


# F19: residual covariance of the ridge transition fit


def _scalar_stats() -> Stats:
    r"""d = 1, V = 2, both tokens unseen: :math:`S_{xx} = 1`, :math:`S_{zx} = 1`, :math:`S_{zz} = 2` on the state
    block. With ridge 1, :math:`W = 1/2`, the residual covariance is :math:`2 - 1/2 - 1/2 + 1/4 = 5/4`, and the
    shortcut :math:`S_{zz} - W S_{zx}^\top` gives 3/2."""
    st = Stats.zeros(1, 1, 2)
    st.pp[:] = 1.0
    st.zp[:] = 1.0
    st.zz[:] = 2.0
    return st


def test_scalar_ridge_residual_covariance() -> None:
    new = m_step(_scalar_stats(), init_params(1, 2, seed=0), ridge=1.0, q_floor=1e-4)
    assert abs(float(new.G[0, 0, 0]) - 0.5) < 1e-12
    assert abs(float(new.Q[0, 0, 0]) - 1.25) < 1e-12


def test_matrix_ridge_residual_covariance_matches_direct_residuals() -> None:
    r"""Moments built from explicit samples (z_prev, y_prev, z); the M-step Q equals the sample mean of
    :math:`(z - W x)(z - W x)^\top` at the returned W, with x = [z_prev, onehot(y_prev)]."""
    rng = np.random.default_rng(0)
    N, d, V, ridge = 400, 3, 4, 0.5
    zp_s = rng.standard_normal((N, d))
    y = rng.integers(0, V, N)
    onehot = np.eye(V)[y]
    z = zp_s @ (0.6 * rng.standard_normal((d, d))) + onehot @ rng.standard_normal((V, d)) + rng.standard_normal((N, d))
    st = Stats.zeros(1, d, V)
    t = lambda a: torch.as_tensor(a, dtype=torch.float64)[None]   # noqa: E731
    st.pp, st.zp, st.zz = t(zp_s.T @ zp_s / N), t(z.T @ zp_s / N), t(z.T @ z / N)
    st.yp, st.yz, st.cnt = t(onehot.T @ zp_s / N), t(onehot.T @ z / N), t(onehot.mean(0))
    new = m_step(st, init_params(d, V, seed=1), ridge=ridge, q_floor=1e-6)
    W = np.concatenate([new.G[0].numpy(), new.E[0].numpy().T], 1)        # (d, d+V)
    x = np.concatenate([zp_s, onehot], 1)                                # (N, d+V)
    r = z - x @ W.T
    direct = r.T @ r / N
    assert np.allclose(new.Q[0].numpy(), direct, atol=1e-10)
    Szx = np.concatenate([st.zp[0].numpy(), st.yz[0].numpy().T], 1)
    shortcut = st.zz[0].numpy() - W @ Szx.T
    assert np.abs(shortcut - direct).max() > 1e-3                        # the ridge term is material here
    assert np.allclose(shortcut - direct, ridge * W @ W.T, atol=1e-10)


def test_residual_covariance_formula() -> None:
    rng = np.random.default_rng(2)
    X = rng.standard_normal((50, 5))
    Z = rng.standard_normal((50, 2))
    W = rng.standard_normal((2, 5))
    R = Z - X @ W.T
    got = residual_covariance(torch.as_tensor(Z.T @ Z), torch.as_tensor(Z.T @ X), torch.as_tensor(X.T @ X),
                              torch.as_tensor(W))
    assert np.allclose(got.numpy(), R.T @ R, atol=1e-10)


# F08: completed-run reuse is bound to the effective configuration


@pytest.fixture
def launcher(monkeypatch: pytest.MonkeyPatch):
    import local_learner_test as llt
    monkeypatch.setattr(llt, "CONFIG", json.loads(json.dumps(llt.CONFIG)))
    llt.CONFIG["stage"] = "B"
    return llt


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_reuse_requires_matching_identity(launcher, tmp_path: Path) -> None:
    llt = launcher
    path = tmp_path / "results_B.jsonl"
    ident = llt.identity(llt.arm_settings("local_w4"), llt.protocol("B"))
    _write(path, [{"stage": "B", "arm": "local_w4", "d": 8, "seed": s, "val_bits": 1.0, "identity": ident}
                  for s in (0, 1)])
    done = llt.done_records(path)
    assert llt.pending(done, "local_w4", 8, [0, 1, 2], ident) == [2]

    llt.CONFIG["local_schedule"] = "rm06" if llt.CONFIG["local_schedule"] != "rm06" else "floor20"
    with pytest.raises(RuntimeError, match="arm_settings.gamma_floor"):
        llt.pending(done, "local_w4", 8, [0, 1, 2], llt.identity(llt.arm_settings("local_w4"), llt.protocol("B")))


@pytest.mark.parametrize("key, value, field", [("backprop_lr", 3e-2, "arm_settings.lr"),
                                               ("B_train_bytes", 1_000, "protocol.B_train_bytes"),
                                               ("pf_particles", 16, "protocol.pf_particles")])
def test_changed_learning_rate_or_protocol_refuses_reuse(launcher, tmp_path: Path, key: str, value: object,
                                                         field: str) -> None:
    llt = launcher
    path = tmp_path / "results_B.jsonl"
    old = llt.identity(llt.arm_settings("backprop"), llt.protocol("B"))
    _write(path, [{"stage": "B", "arm": "backprop", "d": 8, "seed": 0, "identity": old}])
    llt.CONFIG[key] = value
    new = llt.identity(llt.arm_settings("backprop"), llt.protocol("B"))
    with pytest.raises(RuntimeError, match=field.replace(".", r"\.")):
        llt.pending(llt.done_records(path), "backprop", 8, [0], new)


def test_source_change_refuses_reuse(launcher, tmp_path: Path) -> None:
    llt = launcher
    ident = llt.identity(llt.arm_settings("blind"), llt.protocol("B"))
    stale = {**ident, "source": "0" * 64}
    with pytest.raises(RuntimeError, match="source"):
        llt.pending({("blind", 8, 0): {"arm": "blind", "identity": stale}}, "blind", 8, [0], ident)


def test_records_without_identity_are_reused_and_reported(launcher, capsys: pytest.CaptureFixture) -> None:
    llt = launcher
    ident = llt.identity(llt.arm_settings("blind"), llt.protocol("B"))
    assert llt.pending({("blind", 8, 0): {"arm": "blind"}}, "blind", 8, [0, 1], ident) == [1]
    assert "predate stored settings" in capsys.readouterr().out


def test_arm_settings_are_exactly_the_trainer_keywords(launcher) -> None:
    r"""train_arm passes the stored settings to the trainer and nothing else, so the record describes the run."""
    llt = launcher
    for arm, fn in (("backprop", train_backprop), ("local_w4", train_local), ("blind", train_local)):
        kw = llt.arm_settings(arm)
        kw.pop("rule")
        inspect.signature(fn).bind(None, None, None, log=None, **kw)
    assert llt.arm_settings("blind")["blind"] is True and llt.arm_settings("local_w16")["window"] == 16
