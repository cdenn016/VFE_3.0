r"""Table Fisher step of the frame-diagonal family ``gaussian_frame_diagonal`` (2026-10-18).

Under this family the prior of type :math:`v` is :math:`p_v = \mathcal N(\mu_v, U_v\operatorname{diag}(s_v)U_v^\top)`
with :math:`U_v` the layer-0 frame of the type (``pos_phi='none'``), so the Fisher metric in :math:`\mu_v` is
:math:`U_v^{-\top}\operatorname{diag}(1/s_v)U_v^{-1}` and the natural step is
:math:`\Delta\mu_v = -\eta(N/n_v)U_v\operatorname{diag}(s_v)U_v^\top\partial_{\mu_v}\mathcal L`. Pinned here on tiny
float64 models with block_glk frames well away from the identity: the step against a dense float64 solve of
:math:`G^{-1}g` built row by row from the definition, the realized Fisher norm at a binding trust radius, the
invariance of the reference-coordinate step under a passive change of frame, the exact KL of the step inside the
F105 band, the launcher dictionary with this family, and the refusal of positional frames.
"""
from __future__ import annotations

import importlib.util
import math
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest
import torch

from tests.profiles import reference_k8_config
from vfe3.config import VFE3Config
from vfe3.families.frame_gaussian import frame_diagonal_kl
from vfe3.geometry.lie_ops import embed_phi, extract_phi
from vfe3.geometry.matrix_log import logm_principal
from vfe3.model.model import VFEModel
from vfe3.table_optim import TABLE_FISHER_DIAGNOSTIC_KEYS
from vfe3.train import build_optimizer, train_step

F64 = torch.float64
FAMILY = "gaussian_frame_diagonal"
TOKENS = torch.tensor([[1, 2, 2, 5], [1, 1, 3, 7]])            # (B_tot, N): types 1, 2, 3, 5, 7 active
SCORED = torch.tensor(8)


@pytest.fixture(autouse=True)
def _quiet_two_threads():
    torch.set_num_threads(2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _cfg(**over: Any) -> VFE3Config:
    r"""The frozen reference route at K = 8 with the frame-diagonal family inside its envelope, block_glk per-head
    frames of width 4 (no launcher read)."""
    values = dict(family=FAMILY, predictive_readout="kernel", block_mlp_mode="coordinate",
                  block_mlp_covariance="passthrough", decode_mode="diagonal_chunked", norm_type_block="layernorm",
                  untie_decode_bank=True, m_table_update_mode="fisher", gauge_group="block_glk",
                  gauge_tying="per_head", pos_phi="none", rope_mode="none", vocab_size=9, embed_dim=8, n_heads=2,
                  weight_decay=0.0, mu_weight_decay=0.0, sigma_weight_decay=0.0)
    return reference_k8_config(**{**values, **over})


def _model(cfg: VFE3Config, phi_std: float = 0.5) -> VFEModel:
    r"""A float64 model whose frames are far from the identity (cond(U_v) of a few) and whose tables are generic."""
    torch.manual_seed(0)
    model = VFEModel(cfg).double()
    gen = torch.Generator().manual_seed(1)
    pb = model.prior_bank
    with torch.no_grad():
        pb.phi_embed.copy_(phi_std * torch.randn(pb.phi_embed.shape, generator=gen, dtype=F64))
        pb.mu_embed.copy_(torch.randn(pb.mu_embed.shape, generator=gen, dtype=F64))
        pb.sigma_log_embed.copy_(0.4 * torch.randn(pb.sigma_log_embed.shape, generator=gen, dtype=F64))
    return model


def _fill_grads(model: VFEModel, seed: int = 0) -> None:
    g = torch.Generator().manual_seed(seed)
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.randn(p.shape, generator=g, dtype=p.dtype)


def _dense_frames(model: VFEModel) -> torch.Tensor:              # (V, K, K) U_v = exp(phi_v . G), from the definition
    G = model.group.generators.to(F64)                           # (n_gen, K, K)
    return torch.linalg.matrix_exp(embed_phi(model.prior_bank.phi_embed.detach().to(F64), G))


def _step(model: VFEModel, cfg: VFE3Config, factor: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    r"""One optimizer step on the filled gradients; returns (Delta mu, Delta l, diagnostics)."""
    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: factor)
    del scheduler
    pb = model.prior_bank
    before_mu, before_l = pb.mu_embed.detach().clone(), pb.sigma_log_embed.detach().clone()
    optimizer._collect_table_diag = True
    optimizer.table_statistics_provider.set_batch(TOKENS, SCORED)
    optimizer.step()
    return pb.mu_embed.detach() - before_mu, pb.sigma_log_embed.detach() - before_l, optimizer._table_diag


def _active() -> torch.Tensor:                                   # (R,) sorted active types
    return torch.unique(TOKENS)


def test_the_type_frames_are_the_definition_and_far_from_the_identity() -> None:
    model = _model(_cfg())
    rows = _active()
    frame = model.token_type_frames(rows)
    assert frame.block_diagonal and frame.forward.shape == (rows.numel(), 2, 4, 4)       # compact per-head blocks
    U = _dense_frames(model)[rows]                                                        # (R, K, K)
    for h in range(2):
        block = U[:, 4 * h:4 * h + 4, 4 * h:4 * h + 4]
        torch.testing.assert_close(frame.forward[:, h], block, rtol=0.0, atol=1e-12)
    assert float(torch.linalg.cond(U).min()) > 2.0


def test_the_step_is_the_dense_natural_gradient_row_by_row() -> None:
    eta = 0.3
    cfg = _cfg(m_table_fisher_lr=eta, m_table_fisher_trust_radius=1e9)
    model = _model(cfg)
    _fill_grads(model)
    pb = model.prior_bank
    U = _dense_frames(model)                                                              # pre-step frames
    s = pb.sigma_log_embed.detach().exp()
    g_mu, g_l = pb.mu_embed.grad.clone(), pb.sigma_log_embed.grad.clone()
    d_mu, d_l, _ = _step(model, cfg)
    count = torch.bincount(TOKENS.reshape(-1), minlength=cfg.vocab_size)
    for v in range(cfg.vocab_size):
        if count[v] == 0:
            assert torch.equal(d_mu[v], torch.zeros_like(d_mu[v])) and torch.equal(d_l[v], torch.zeros_like(d_l[v]))
            continue
        U_inv = torch.linalg.inv(U[v])
        G_mu = U_inv.mT @ torch.diag(1.0 / s[v]) @ U_inv                                  # (K, K) Fisher of p_v in mu_v
        weight = float(SCORED) / float(count[v])
        torch.testing.assert_close(d_mu[v], -eta * weight * torch.linalg.solve(G_mu, g_mu[v]), rtol=0.0, atol=1e-10)
        torch.testing.assert_close(d_l[v], -eta * weight * 2.0 * g_l[v], rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("factor", [1.0, 0.5])
def test_the_realized_fisher_norm_is_the_scheduled_radius_when_the_trust_binds(factor: float) -> None:
    radius = 0.05
    cfg = _cfg(m_table_fisher_lr=10.0, m_table_fisher_trust_radius=radius)
    model = _model(cfg)
    _fill_grads(model)
    U = _dense_frames(model)
    s = model.prior_bank.sigma_log_embed.detach().exp()
    d_mu, d_l, diag = _step(model, cfg, factor)
    rows = _active()
    xi = torch.linalg.solve(U[rows], d_mu[rows].unsqueeze(-1)).squeeze(-1)               # (R, K) U^-1 Delta mu
    norm = ((xi.square() / s[rows]).sum(-1) + 0.5 * d_l[rows].square().sum(-1)).sqrt()
    torch.testing.assert_close(norm, torch.full_like(norm, factor * radius), rtol=1e-10, atol=0.0)
    assert set(diag) == set(TABLE_FISHER_DIAGNOSTIC_KEYS) and diag["table_fisher_active_rows"] == rows.numel()
    assert diag["table_fisher_trust_scale_mean"] < 1.0
    assert math.isclose(diag["table_fisher_step_norm_mean"], factor * radius, rel_tol=1e-10)


def test_the_reference_step_is_invariant_under_a_passive_change_of_frame() -> None:
    r"""With :math:`U_v \to g_vU_v`, :math:`\mu_v \to g_v\mu_v` and the covariant gradient :math:`g_v^{-\top}\partial_\mu`,
    the step in :math:`\xi_v = U_v^{-1}\mu_v` and the log-variance step do not move."""
    cfg = _cfg(m_table_fisher_lr=0.7, m_table_fisher_trust_radius=0.2)
    G = None
    results = []
    for change in (False, True):
        model = _model(cfg)
        _fill_grads(model)
        pb = model.prior_bank
        G = model.group.generators.to(F64)
        U = _dense_frames(model)                                                          # (V, K, K)
        if change:
            a = torch.randn(cfg.vocab_size, G.shape[0], generator=torch.Generator().manual_seed(5), dtype=F64)
            g = torch.linalg.matrix_exp(0.3 * embed_phi(a, G))                            # (V, K, K) g_v in the group
            log_gU, failed, _ = logm_principal(g @ U)
            assert not bool(failed.any())
            with torch.no_grad():
                pb.phi_embed.copy_(extract_phi(log_gU, G))
                pb.mu_embed.copy_((g @ pb.mu_embed.unsqueeze(-1)).squeeze(-1))
                pb.mu_embed.grad = torch.linalg.solve(g.mT, pb.mu_embed.grad.unsqueeze(-1)).squeeze(-1)
            U = _dense_frames(model)
            torch.testing.assert_close(U, g @ _dense_frames(_model(cfg)), rtol=0.0, atol=1e-11)
        d_mu, d_l, _ = _step(model, cfg)
        results.append((torch.linalg.solve(U, d_mu.unsqueeze(-1)).squeeze(-1), d_l))
    (xi0, l0), (xi1, l1) = results
    assert float(xi0.abs().max()) > 1e-3
    torch.testing.assert_close(xi1, xi0, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(l1, l0, rtol=0.0, atol=1e-14)


@pytest.mark.parametrize("radius", [0.02, 0.1])
def test_the_exact_kl_of_the_step_stays_in_the_f105_band(radius: float) -> None:
    r""":math:`\mathrm{KL}(p_v^{+}\|p_v)/(\tfrac12\|\Delta\|_G^2) \in [f(-x), f(x)]` with :math:`f(x) = 2(e^x-1-x)/x^2`
    and :math:`x = \sqrt2\|\Delta\|_G` (the mean part is exact; each log-variance coordinate obeys the one-coordinate
    bound of audit 2026-10-16 F105)."""
    cfg = _cfg(m_table_fisher_lr=10.0, m_table_fisher_trust_radius=radius)
    model = _model(cfg)
    _fill_grads(model)
    pb = model.prior_bank
    rows = _active()
    U = _dense_frames(model)[rows]
    mu0, l0 = pb.mu_embed.detach()[rows].clone(), pb.sigma_log_embed.detach()[rows].clone()
    d_mu, d_l, _ = _step(model, cfg)
    mu1, l1 = mu0 + d_mu[rows], l0 + d_l[rows]
    divergence = frame_diagonal_kl(mu1, l1.exp(), U, mu0, l0.exp(), U)                    # (R,) exact KL(p+ || p)
    xi = torch.linalg.solve(U, d_mu[rows].unsqueeze(-1)).squeeze(-1)
    norm_sq = (xi.square() / l0.exp()).sum(-1) + 0.5 * d_l[rows].square().sum(-1)
    torch.testing.assert_close(norm_sq.sqrt(), torch.full_like(norm_sq, radius), rtol=1e-10, atol=0.0)
    x = math.sqrt(2.0) * radius
    band = (2.0 * (math.exp(-x) - 1.0 + x) / x ** 2, 2.0 * (math.exp(x) - 1.0 - x) / x ** 2)
    ratio = divergence / (0.5 * norm_sq)
    assert bool((ratio >= band[0] - 1e-9).all()) and bool((ratio <= band[1] + 1e-9).all()), (ratio, band)


def test_the_diagonal_step_is_taken_on_identity_frames_and_other_families() -> None:
    off = build_optimizer(VFEModel(_cfg(gauge_transport="off")), _cfg(gauge_transport="off"))
    assert not off.table_statistics_provider.reads_frames
    diagonal = _cfg(family="gaussian_diagonal")
    assert not build_optimizer(VFEModel(diagonal), diagonal).table_statistics_provider.reads_frames
    assert build_optimizer(VFEModel(_cfg()), _cfg()).table_statistics_provider.reads_frames


@pytest.mark.parametrize("pos_phi", ["learned", "frozen"])
def test_positional_frames_are_refused_with_the_reason(pos_phi: str) -> None:
    with pytest.raises(ValueError, match=f"m_table_update_mode='fisher' with pos_phi='{pos_phi}'"):
        _cfg(pos_phi=pos_phi)


def _launcher(path: Path) -> Dict[str, Any]:
    r"""The ``config`` dictionary of the launcher at ``path``, loaded by file path (its ``__main__`` block never runs)."""
    spec = importlib.util.spec_from_file_location("_launcher_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(module.config)


_TINY_SHAPES = dict(vocab_size=11, embed_dim=8, n_heads=2, max_seq_len=4, batch_size=2, n_layers=2,
                    block_mlp_expansion=2, eval_stride=None)


def test_the_launcher_dictionary_with_this_family_trains() -> None:
    r"""The launcher at ``VFE3_LAUNCHER_PATH`` (default: this checkout's ``train_vfe3.py``) with ``family`` set to this
    family and only the shape fields shrunk constructs and takes finite training steps; a launcher route outside the
    family's envelope for another reason is skipped with that reason.

    The toggles the family's envelope couples to it are set with it, whatever the launcher holds: the kernel readout,
    the coordinate block MLP, a linear or diagonal KL decoder, and token-only frames (``pos_phi='none'``) under the
    Fisher table step."""
    path = Path(os.environ.get("VFE3_LAUNCHER_PATH", Path(__file__).resolve().parents[1] / "train_vfe3.py"))
    launcher = _launcher(path)
    envelope: Dict[str, Any] = {"family": FAMILY, "predictive_readout": "kernel", "block_mlp_mode": "coordinate"}
    if launcher.get("decode_mode") not in ("linear", "diagonal_chunked"):
        envelope["decode_mode"] = "diagonal_chunked"
    if launcher.get("m_table_update_mode") == "fisher":
        envelope["pos_phi"] = "none"
    values = {**launcher, **envelope, **_TINY_SHAPES}
    if values.get("kl_max") is not None:
        values["kl_max"] = float(values["kl_max"]) * 8 / _launcher(path)["embed_dim"]
    try:
        cfg = VFE3Config(**values)
    except ValueError as error:
        assert "m_table_update_mode" not in str(error), error
        pytest.skip(f"the launcher route is outside the family's envelope for another reason: {error}")
    torch.manual_seed(0)
    model = VFEModel(cfg)
    optimizer = build_optimizer(model, cfg)
    assert optimizer.table_statistics_provider.reads_frames == (cfg.m_table_update_mode == "fisher"
                                                                and cfg.gauge_transport != "off")
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    gen = torch.Generator().manual_seed(4)
    for _ in range(3):
        tokens = torch.randint(0, cfg.vocab_size, (2, 4), generator=gen)
        loss = float(train_step(model, optimizer, scheduler, tokens, torch.roll(tokens, -1, dims=1), grad_clip=1.0))
        assert math.isfinite(loss)


def test_a_few_training_steps_with_the_frame_fisher_table_keep_the_loss_finite() -> None:
    cfg = _cfg()
    torch.manual_seed(0)
    model = VFEModel(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    gen = torch.Generator().manual_seed(4)
    mu_before = model.prior_bank.mu_embed.detach().clone()
    for _ in range(4):
        tokens = torch.randint(0, cfg.vocab_size, (2, 4), generator=gen)
        loss = float(train_step(model, optimizer, scheduler, tokens, torch.roll(tokens, -1, dims=1), grad_clip=1.0))
        assert math.isfinite(loss)
    assert not torch.equal(model.prior_bank.mu_embed.detach(), mu_before)
