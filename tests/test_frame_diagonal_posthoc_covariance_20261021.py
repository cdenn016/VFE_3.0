r"""Post-hoc and semantics routing covariances under ``gaussian_frame_diagonal`` (audit 2026-10-20 item 26).

The family stores the reference variances :math:`c_i` with :math:`\Sigma_i = U_i\operatorname{diag}(c_i)U_i^\top`, so the
routing covariance is :math:`C_i = U_i^{-1}\Sigma_iU_i^{-\top} = \operatorname{diag}(c_i)`, its log-determinant is
:math:`\sum_k\log c_{ik}` and its frame part is zero. Before the fix both readers formed
:math:`U_i^{-1}\operatorname{diag}(c_i)U_i^{-\top}`. Tiny CPU models and synthetic tables only.
"""
import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from tests.profiles import live_config
from vfe3.model.model import VFEModel
from vfe3.posthoc import covariance as cv
from vfe3.semantics import frames
from vfe3.semantics.run import SemanticRun

F64 = torch.float64


def _frame_diagonal_model() -> VFEModel:
    cfg = live_config(family="gaussian_frame_diagonal", predictive_readout="kernel", block_mlp_mode="coordinate",
                      block_mlp_covariance="passthrough", decode_mode="diagonal_chunked", norm_type_block="layernorm",
                      untie_decode_bank=True)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():                                      # frames far from orthogonal, distinct variances
        model.prior_bank.phi_embed.copy_(0.4 * torch.randn(model.prior_bank.phi_embed.shape, generator=g))
        model.prior_bank.sigma_log_embed.copy_(torch.randn(model.prior_bank.sigma_log_embed.shape, generator=g))
    return model


def test_posthoc_routing_covariance_is_the_reference_diagonal() -> None:
    model = _frame_diagonal_model()
    assert cv.stores_reference_variances(model) and cv.gap_applicability(model) is None
    inputs = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    keep = torch.ones(inputs.shape, dtype=torch.bool)
    inverse, sigma, _ = cv.routing_frame_and_covariance(model, inputs)
    G = inverse.double() @ inverse.double().transpose(-1, -2)
    assert float((G - torch.eye(G.shape[-1], dtype=F64)).abs().max()) > 0.1     # frames matter if misread
    stats, _ = cv._sample_statistics(model, [(inputs, keep)], torch.device("cpu"))
    c = sigma.double().reshape(-1, sigma.shape[-1])                                # (P, K) reference variances
    np.testing.assert_allclose(stats["cc"], c.square().sum(-1).numpy(), rtol=1e-12)
    np.testing.assert_allclose(stats["ldC"], c.log().sum(-1).numpy(), rtol=1e-10, atol=1e-10)
    # The pre-fix reading U^{-1} diag(c) U^{-T} differs on this model.
    misread = cv.pair_statistics(inverse, sigma, keep)
    assert float(np.abs(misread["cc"] - stats["cc"]).max()) > 1e-3 * float(stats["cc"].max())


class _Ctx:
    def __init__(self, n: int) -> None:
        self._log_count = np.linspace(0.0, 3.0, n)

    def log_count(self, ids: Any) -> np.ndarray:
        return self._log_count[: len(ids)]


def _fake_run(family: str, n: int, B: int, d: int) -> SimpleNamespace:
    g = np.random.default_rng(0)
    lv = g.normal(size=(n, B * d))
    X = [0.3 * g.normal(size=(n, d, d)) for _ in range(B)]
    return SimpleNamespace(cfg=SimpleNamespace(family=family), reference_variances=SemanticRun.reference_variances.fget(
                               SimpleNamespace(cfg=SimpleNamespace(family=family))), block_slices=[slice(b * d, (b + 1) * d) for b in range(B)],
                           log_var=lambda ids: lv[np.asarray(ids)], phi_blocks=lambda ids: [x[np.asarray(ids)] for x in X],
                           _lv=lv, _X=X)


def test_semantics_routing_covariance_has_no_frame_part_under_the_family() -> None:
    n, B, d = 12, 2, 3
    run = _fake_run("gaussian_frame_diagonal", n, B, d)
    out = frames.routing_covariance_summary(run, _Ctx(n), np.arange(n))
    np.testing.assert_allclose(out["log_det_C"], run._lv.sum(1), rtol=1e-12, atol=1e-12)
    assert np.all(out["frame_part"] == 0.0)
    assert out["identity_max_abs_error"] < 1e-10
    assert out["variance_shares"] == {"sigma": 1.0, "frame": 0.0, "cross": 0.0}
    assert "diag(c_v)" in out["covariance"]
    # The local diagonal family keeps the frame part -2 tr X.
    local = frames.routing_covariance_summary(_fake_run("gaussian_diagonal", n, B, d), _Ctx(n), np.arange(n))
    assert float(np.abs(local["frame_part"]).max()) > 1e-3


def test_semantics_log_det_adds_the_type_frame_under_the_family() -> None:
    n, B, d = 5, 2, 3
    fake = _fake_run("gaussian_frame_diagonal", n, B, d)
    fake.reference_variances = True
    fake.state = {"prior_bank.phi_embed": None}
    log_det = SemanticRun.log_det(fake, np.arange(n))
    trace = sum(np.trace(x, axis1=-2, axis2=-1) for x in fake._X)
    np.testing.assert_allclose(log_det, fake._lv.sum(1) + 2.0 * trace, rtol=1e-12)
    # det Sigma = det(e^X)^2 prod c against a dense evaluation of one block-diagonal Sigma_v.
    U = torch.block_diag(*(torch.matrix_exp(torch.from_numpy(x[0])) for x in fake._X))
    Sigma = U @ torch.diag(torch.exp(torch.from_numpy(fake._lv[0]))) @ U.T
    assert math.isclose(float(torch.linalg.slogdet(Sigma).logabsdet), float(log_det[0]), rel_tol=1e-10)


@pytest.mark.parametrize("family, expected", [("gaussian_frame_diagonal", True), ("gaussian_diagonal", False)])
def test_semantic_run_flags_reference_variances(family: str, expected: bool) -> None:
    fake = SimpleNamespace(cfg=SimpleNamespace(family=family))
    assert SemanticRun.reference_variances.fget(fake) is expected
