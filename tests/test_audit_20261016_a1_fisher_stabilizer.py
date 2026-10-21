r"""Stabilizer-only completion of the belief-Fisher frame metric (audit 2026-10-16, F50).

The frame Fisher :math:`g(Z) = \|Z\eta\|^2 + 2\|\mathrm{sym}\,Z\|_F^2` (whitened, :math:`Z = L^{-1} X L`,
:math:`\eta = L^{-1}\xi`) vanishes on the stabilizer :math:`\{Z : \mathrm{sym}\,Z = 0,\ Z\eta = 0\}`. The shipped
completion ``m_phi_fisher_completion='rotation'`` adds :math:`\lambda\|\mathrm{asym}\,Z\|_F^2` on every rotation, so
it also raises the metric of the Fisher-visible rotations (those moving :math:`\eta`) and no
``m_phi_fisher_rotation_weight`` recovers the belief-Fisher natural gradient there. ``'stabilizer'`` adds
:math:`\lambda\|P_\perp\,\mathrm{asym}(Z)\,P_\perp\|_F^2` only (:math:`P_\perp = I - nn^\top`, :math:`n = \eta/|\eta|`),
the orthogonal projector onto that kernel, so the whitened solve is :math:`F^+ R + \Pi R/\lambda`: the pseudo-inverse
natural gradient on the Fisher-visible subspace plus a kernel step that vanishes as :math:`\lambda \to \infty`.
Token-type rows only; positional rows are refused. Tiny float64 checks against dense operators.
"""
from __future__ import annotations

import functools
import math
import warnings

import pytest
import torch
from torch import nn

from tests.profiles import config_values, test_config
from vfe3.config import VFE3Config
from vfe3.gauge_optim import GaugeManifoldAdamW, belief_fisher_group_direction, stage_group_log_candidate
from vfe3.geometry.groups import get_group
from vfe3.geometry.phi_fisher import (
    FisherRowStatistics,
    fisher_spectrum_bounds,
    fisher_type_solve,
    type_metric_residual,
    type_row_spectrum,
)

F64 = torch.float64


def _whitened_operator(eta: torch.Tensor, lam: float, completion: str) -> torch.Tensor:
    r"""Row-major (d^2, d^2) matrix of :math:`Z \mapsto Z\eta\eta^\top + 2\,\mathrm{sym}Z + \lambda\,\Pi Z`."""
    d = eta.shape[-1]
    norm = eta.norm()
    n = eta / norm if float(norm) > 0 else torch.zeros_like(eta)
    P = torch.eye(d, dtype=F64) - torch.outer(n, n)

    def apply(Z: torch.Tensor) -> torch.Tensor:
        asym = 0.5 * (Z - Z.T)
        penalty = asym if completion == "rotation" else P @ asym @ P
        return Z @ torch.outer(eta, eta) + (Z + Z.T) + lam * penalty

    basis = torch.eye(d * d, dtype=F64).reshape(d * d, d, d)
    return torch.stack([apply(E).reshape(-1) for E in basis], dim=1)


def _row(d: int, seed: int, mean_scale: float = 0.4):
    gen = torch.Generator().manual_seed(seed)
    mu = mean_scale * torch.randn(1, 1, d, generator=gen, dtype=F64)
    s = torch.exp(0.4 * torch.randn(1, 1, d, generator=gen, dtype=F64))
    G = torch.randn(1, 1, d, d, generator=gen, dtype=F64)
    return mu, s, G


@pytest.mark.parametrize("d", [3, 4, 5])
@pytest.mark.parametrize("lam", [0.5, 2.0, 50.0])
def test_stabilizer_solve_is_the_pseudo_inverse_on_the_visible_subspace(d: int, lam: float) -> None:
    mu, s, G = _row(d, seed=10 + d)
    scale = torch.tensor([1.7], dtype=F64)
    r = s[0, 0].sqrt()
    ratio = r[:, None] / r[None, :]
    R = (scale[0] * G[0, 0] * ratio).reshape(-1)                            # whitened covector
    eta = mu[0, 0] / r
    F0 = _whitened_operator(eta, 0.0, "rotation")                            # the belief Fisher itself
    eigenvalues, V = torch.linalg.eigh(F0)
    kernel = eigenvalues.abs() <= 1e-10 * eigenvalues.abs().max()
    assert int(kernel.sum()) == (d - 1) * (d - 2) // 2
    visible, null = V[:, ~kernel], V[:, kernel]
    pinv = torch.linalg.pinv(F0, rtol=1e-10) @ R

    def whitened(W: torch.Tensor) -> torch.Tensor:
        return (W[0, 0] / ratio).reshape(-1)

    Z = whitened(fisher_type_solve(G, mu, s, scale, rotation_weight=lam, completion="stabilizer"))
    deviation = (visible.T @ (Z - pinv)).norm() / (visible.T @ pinv).norm()
    assert float(deviation) <= 1e-12
    torch.testing.assert_close(null.T @ Z, null.T @ R / lam, rtol=1e-10, atol=1e-13)
    dense = torch.linalg.solve(_whitened_operator(eta, lam, "stabilizer"), R)
    torch.testing.assert_close(Z, dense, rtol=1e-10, atol=1e-12)
    # The shipped rotation completion is the dense rotation solve and is NOT the pseudo-inverse there.
    Z_rot = whitened(fisher_type_solve(G, mu, s, scale, rotation_weight=lam))
    torch.testing.assert_close(Z_rot, torch.linalg.solve(_whitened_operator(eta, lam, "rotation"), R),
                               rtol=1e-10, atol=1e-12)
    assert float((visible.T @ (Z_rot - pinv)).norm() / (visible.T @ pinv).norm()) > 1e-2


@pytest.mark.parametrize("lam", [0.5, 3.0])
def test_zero_mean_rows_coincide_with_the_rotation_completion(lam: float) -> None:
    _, s, G = _row(4, seed=3)
    mu = torch.zeros(1, 1, 4, dtype=F64)                                     # stabilizer = every rotation
    scale = torch.ones(1, dtype=F64)
    torch.testing.assert_close(
        fisher_type_solve(G, mu, s, scale, rotation_weight=lam, completion="stabilizer"),
        fisher_type_solve(G, mu, s, scale, rotation_weight=lam), rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("d", [1, 2, 3, 4])
@pytest.mark.parametrize("mean_scale", [0.0, 0.05, 1.5])
@pytest.mark.parametrize("lam", [0.5, 2.0, 7.0])
def test_stabilizer_spectrum_matches_the_dense_operator(d: int, mean_scale: float, lam: float) -> None:
    mu, s, _ = _row(d, seed=40 + d, mean_scale=mean_scale)
    eta = mu[0, 0] / s[0, 0].sqrt()
    eigenvalues = torch.linalg.eigvalsh(_whitened_operator(eta, lam, "stabilizer"))
    a = (mu.square() / s).sum(dim=-1)                                        # (1, 1)
    lam_min, lam_max = fisher_spectrum_bounds(a, d, rotation_weight=lam, completion="stabilizer")
    assert abs(float(lam_min) - float(eigenvalues[0])) <= 1e-11
    assert abs(float(lam_max) - float(eigenvalues[-1])) <= 1e-11
    a2, lam_min2, cond = type_row_spectrum(mu, s, rotation_weight=lam, completion="stabilizer")
    torch.testing.assert_close(a2, a)
    torch.testing.assert_close(cond, lam_max / lam_min)


def test_metric_residual_of_the_stabilizer_solve_vanishes() -> None:
    gen = torch.Generator().manual_seed(5)
    mu = 0.5 * torch.randn(3, 2, 4, generator=gen, dtype=F64)
    mu[1, 0] = 0.0                                                            # one zero-mean block
    s = torch.exp(0.3 * torch.randn(3, 2, 4, generator=gen, dtype=F64))
    G = torch.randn(3, 2, 4, 4, generator=gen, dtype=F64)
    scale = torch.tensor([1.0, 2.0, 0.5], dtype=F64)
    W = fisher_type_solve(G, mu, s, scale, rotation_weight=1.5, completion="stabilizer")
    residual = type_metric_residual(W, G, mu, s, scale, rotation_weight=1.5, completion="stabilizer")
    assert float(residual.max()) <= 1e-12
    wrong = type_metric_residual(W, G, mu, s, scale, rotation_weight=1.5)    # read against the rotation metric
    assert float(wrong.max()) > 1e-3


def test_unknown_completion_and_positional_rows_are_refused() -> None:
    mu, s, G = _row(3, seed=1)
    with pytest.raises(ValueError, match="completion"):
        fisher_type_solve(G, mu, s, torch.ones(1, dtype=F64), rotation_weight=2.0, completion="kernel")
    group = get_group("block_glk")(6, 2)
    with pytest.raises(ValueError, match="completion"):
        GaugeManifoldAdamW([nn.Parameter(torch.zeros(1))], group, phi_group_trust_radius=0.1,
                           phi_chart_max_norm=5.0, phi_fisher_completion="kernel")
    rows = 2
    pos = FisherRowStatistics(kind="pos", count=torch.ones(rows, dtype=F64), total=torch.tensor(4.0, dtype=F64),
                              pooled_operator=torch.eye(9, dtype=F64).expand(rows, 2, 9, 9).clone())
    phi = torch.zeros(rows, 18, dtype=F64)
    U = torch.eye(3, dtype=F64).expand(rows, 2, 3, 3).clone()
    with pytest.raises(ValueError, match="stabilizer"):
        belief_fisher_group_direction(torch.randn(rows, 18, dtype=F64), phi, U, collect=False,
                                      statistics=pos, rotation_weight=2.0, completion="stabilizer")


def _stub_provider(mu: torch.Tensor, s: torch.Tensor, counts: torch.Tensor):
    def provider(param: nn.Parameter, rows: torch.Tensor) -> FisherRowStatistics:
        return FisherRowStatistics(kind="type", count=counts[rows], total=torch.tensor(8.0, dtype=F64),
                                   local_mean=mu[rows], local_var=s[rows])
    return provider


def test_optimizer_steps_with_the_selected_completion() -> None:
    group = get_group("block_glk")(6, 2)
    gen = torch.Generator().manual_seed(9)
    phi0 = 0.2 * torch.randn(4, 18, generator=gen, dtype=F64)
    grad = torch.randn(4, 18, generator=gen, dtype=F64)
    mu = 0.6 * torch.randn(4, 2, 3, generator=gen, dtype=F64)
    s = torch.exp(0.3 * torch.randn(4, 2, 3, generator=gen, dtype=F64))
    counts = torch.tensor([2.0, 1.0, 3.0, 1.0], dtype=F64)
    provider = _stub_provider(mu, s, counts)
    after = {}
    for completion in ("rotation", "stabilizer"):
        phi = nn.Parameter(phi0.clone())
        optimizer = GaugeManifoldAdamW(
            [{"params": [phi], "lr": 0.05, "pullback_group": True, "weight_decay": 0.0}],
            group, phi_group_trust_radius=10.0, phi_chart_max_norm=5.0, phi_group_retraction="group_log",
            phi_statistics_provider=provider, phi_fisher_rotation_weight=1.5,
            phi_fisher_completion=completion, weight_decay=0.0)
        phi.grad = grad.clone()
        optimizer.step()
        expected = stage_group_log_candidate(
            grad, phi0, group,
            direction=functools.partial(belief_fisher_group_direction, statistics=provider(phi, torch.arange(4)),
                                        rotation_weight=1.5, completion=completion),
            learning_rate=0.05, trust_radius=10.0, chart_max_norm=5.0).candidate_phi
        torch.testing.assert_close(phi.detach(), expected, rtol=0, atol=0)
        after[completion] = phi.detach().clone()
    assert not torch.allclose(after["rotation"], after["stabilizer"])


def _fisher_cfg(**overrides) -> VFE3Config:
    values = dict(vocab_size=12, embed_dim=6, n_heads=2, max_seq_len=8, n_layers=1, gauge_group="block_glk",
                  pos_phi="none", m_phi_update_mode="fisher_group", transport_chart_max_norm=6.0, m_phi_lr=0.1)
    values.update(overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return test_config(**values)


@pytest.mark.parametrize("retraction", ["group_log", "chart"])
def test_tiny_model_train_step_under_the_stabilizer_completion(retraction: str) -> None:
    from vfe3.model.model import VFEModel
    from vfe3.train import build_optimizer, train_step

    cfg = _fisher_cfg(m_phi_fisher_completion="stabilizer", m_phi_group_retraction=retraction)
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(cfg)
        optimizer = build_optimizer(model, cfg)
    assert optimizer._phi_fisher_completion == "stabilizer"            # build_optimizer forwards the field
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    tokens = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3], [3, 2, 1, 0, 3, 2, 1, 0]])
    phi_before = model.prior_bank.phi_embed.detach().clone()
    optimizer._collect_gauge_diag = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loss = train_step(model, optimizer, scheduler, tokens, torch.roll(tokens, -1, dims=1), grad_clip=1.0)
    assert math.isfinite(float(loss))
    phi_after = model.prior_bank.phi_embed.detach()
    assert torch.equal(phi_after[4:], phi_before[4:])                         # types absent from the batch
    assert not torch.equal(phi_after[:4], phi_before[:4])
    assert optimizer._gauge_diag["phi_group_rejected_rows"] == 0.0
    assert optimizer._gauge_diag["phi_fisher_condP_max"] >= 1.0


def test_config_field_default_validation_and_inactive_marks() -> None:
    # The test baseline pins the rotation completion; the VFE3Config default only has to be admitted.
    assert config_values()["m_phi_fisher_completion"] == "rotation"
    default = VFE3Config.__dataclass_fields__["m_phi_fisher_completion"].default
    assert _fisher_cfg(m_phi_fisher_completion=default).m_phi_fisher_completion == default
    assert _fisher_cfg(m_phi_fisher_completion="stabilizer").m_phi_fisher_completion == "stabilizer"
    assert _fisher_cfg(m_phi_fisher_completion="stabilizer", pos_phi="frozen").pos_phi == "frozen"
    with pytest.raises(ValueError, match="m_phi_fisher_completion"):
        _fisher_cfg(m_phi_fisher_completion="kernel")
    with pytest.raises(ValueError, match="m_phi_fisher_completion='stabilizer'"):
        _fisher_cfg(m_phi_fisher_completion="stabilizer", pos_phi="learned")
    assert "m_phi_fisher_completion" not in _fisher_cfg().inactive_fields()
    assert "m_phi_fisher_completion" in test_config(m_phi_update_mode="adamw").inactive_fields()
    assert math.isfinite(_fisher_cfg().m_phi_fisher_rotation_weight)
