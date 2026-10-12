r"""Audit 2026-10-12 round 2, item 14 and M4: the frame condition bound shares its budget and leaves a record.

With :math:`L = \log\kappa_{\max}`, a learned positional row is bounded to spread :math:`L/2`, the realized positional
spread :math:`s_h = \max_n \operatorname{spread}(S_0(Y_n)_h)` is measured, and the token factor receives
:math:`L - s_h`, so :math:`\kappa_2(e^{X_v} e^{Y_n})_h \le \kappa_{\max}` for every (token, position, head), including
a fixed ``pos_phi='frozen'`` factor. Projected rows carry a storage margin, so the bound is a fixed point; it runs at
construction, on checkpoint load and after each accepted step, and its row counts reach the S1 panel D counters.
Tiny models only (K = 6, two heads of d = 3).
"""
import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
from torch.utils.data import DataLoader

import vfe3.train as vt
from tests.profiles import test_config
from vfe3.data.datasets import TokenWindows
from vfe3.gauge_optim import _frame_bound_rows, bound_phi_frame_condition_
from vfe3.model.model import VFEModel
from vfe3.run_artifacts import RunArtifacts, _pure_path_report

KAPPA = 20.0


def _cfg(**overrides: Any) -> Any:
    values = dict(vocab_size=8, embed_dim=6, n_heads=2, max_seq_len=4, n_layers=1, n_e_steps=1,
                  gauge_group="block_glk", pos_phi="learned", warmup_steps=1, max_steps=4, batch_size=2)
    values.update(overrides)
    return test_config(**values)


def _blocks(phi: torch.Tensor, model: VFEModel) -> torch.Tensor:          # (rows, H, d, d) float64
    group = model.group
    M = torch.einsum("ra,aij->rij", phi.detach().to(torch.float64), group.generators.to(torch.float64))
    d = group.irrep_dims[0]
    return torch.stack([M[:, h * d:(h + 1) * d, h * d:(h + 1) * d] for h in range(len(group.irrep_dims))], dim=1)


def _spread(blocks: torch.Tensor) -> torch.Tensor:                         # (rows, H)
    S = 0.5 * (blocks + blocks.transpose(-1, -2))
    d = blocks.shape[-1]
    S0 = S - S.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None] * torch.eye(d, dtype=S.dtype)
    lam = torch.linalg.eigvalsh(S0)
    return lam[..., -1] - lam[..., 0]


def _composed_condition(model: VFEModel) -> torch.Tensor:                 # (V, N, H): every (token, position, head)
    tokens = torch.linalg.matrix_exp(_blocks(model.prior_bank.phi_embed, model))     # (V, H, d, d)
    template = torch.zeros(model.cfg.max_seq_len, model.group.generator_shape[0])
    Y = model._pos_phi_right(template)
    positions = torch.linalg.matrix_exp(_blocks(Y, model))                            # (N, H, d, d)
    return torch.linalg.cond(tokens[:, None] @ positions[None, :])


def _stretch(model: VFEModel, *, token_scale: float, pos_scale: float, seed: int = 0) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        phi = model.prior_bank.phi_embed
        phi.copy_(token_scale * torch.randn(phi.shape, generator=generator))
        if getattr(model, "pos_phi_free", None) is not None:
            model.pos_phi_free.copy_(pos_scale * torch.randn(model.pos_phi_free.shape, generator=generator))


def test_composed_condition_is_bounded_for_every_token_position_and_head_after_a_step() -> None:
    torch.manual_seed(0)
    cfg = _cfg(phi_frame_condition_max=KAPPA)
    model = VFEModel(cfg)
    _stretch(model, token_scale=1.5, pos_scale=1.0)
    assert float(_composed_condition(model).max()) > 5 * KAPPA
    optimizer = vt.build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    vt.train_step(model, optimizer, scheduler, torch.tensor([[0, 1, 2, 3]]), torch.tensor([[1, 2, 3, 4]]))
    assert float(_composed_condition(model).max()) <= KAPPA * (1 + 1e-5)
    assert float(_spread(_blocks(model.pos_phi_free, model)).max()) <= 0.5 * math.log(KAPPA) * (1 + 1e-6)


def test_adaptive_budget_distorts_the_token_table_less_than_the_even_split() -> None:
    model = VFEModel(_cfg(phi_frame_condition_max=KAPPA))
    _stretch(model, token_scale=1.5, pos_scale=0.01)                  # positional rows use little of their half
    before = model.prior_bank.phi_embed.detach().clone()
    # The even split: every factor bounded to spread L/2, whatever the positional rows use.
    even, over, _ = _frame_bound_rows(before, None, None, 2, 3, math.log(KAPPA) / 4, torch.float64,
                                      store_dtype=before.dtype)
    even = torch.where(over[:, None], even.to(before.dtype), before)
    bound_phi_frame_condition_(model, KAPPA)
    adaptive_change = float((model.prior_bank.phi_embed.detach() - before).norm())
    even_change = float((even - before).norm())
    assert 0.0 < adaptive_change < 0.75 * even_change
    assert float(_composed_condition(model).max()) <= KAPPA * (1 + 1e-5)


def test_frozen_positional_factor_is_inside_the_budget() -> None:
    model = VFEModel(_cfg(pos_phi="frozen", pos_phi_scale=0.3, phi_frame_condition_max=KAPPA))
    positional = _blocks(model._pos_phi_right(torch.zeros(4, model.group.generator_shape[0])), model)
    assert float(_spread(positional).max()) > 0.25 * math.log(KAPPA)  # the fixed factor uses real budget
    _stretch(model, token_scale=1.5, pos_scale=0.0)
    assert float(_composed_condition(model).max()) > 5 * KAPPA
    bound_phi_frame_condition_(model, KAPPA)
    assert float(_composed_condition(model).max()) <= KAPPA * (1 + 1e-5)


def test_frozen_factor_over_the_whole_budget_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="fixed positional factor"):
        VFEModel(_cfg(pos_phi="frozen", pos_phi_scale=5.0, phi_frame_condition_max=KAPPA))


def test_construction_bounds_the_initial_frames() -> None:
    torch.manual_seed(0)
    unbounded = VFEModel(_cfg(phi_scale=1.5, pos_phi_scale=1.0))
    assert float(_composed_condition(unbounded).max()) > KAPPA
    torch.manual_seed(0)
    bounded = VFEModel(_cfg(phi_scale=1.5, pos_phi_scale=1.0, phi_frame_condition_max=KAPPA))
    assert float(_composed_condition(bounded).max()) <= KAPPA * (1 + 1e-5)
    # The logged vertex conditioning is the head-block condition the bound controls.
    tokens = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    assert unbounded.diagnostics(tokens)["vertex_cond_max"] > KAPPA
    assert bounded.diagnostics(tokens)["vertex_cond_max"] <= KAPPA * (1 + 1e-5)


@pytest.mark.parametrize("overrides", [
    dict(gauge_group="block_glk", embed_dim=6, n_heads=2),
    dict(gauge_group="block_glk", embed_dim=6, n_heads=2, gauge_tying="shared"),
    dict(gauge_group="block_glmc", embed_dim=8, n_heads=2),
])
def test_projection_is_a_bitwise_fixed_point(overrides: dict) -> None:
    r"""Rows with a large skew part stress the storage roundoff; ten further applications change nothing."""
    model = VFEModel(_cfg(phi_frame_condition_max=KAPPA, **overrides))
    generator = torch.Generator().manual_seed(3)
    with torch.no_grad():
        phi = model.prior_bank.phi_embed
        phi.copy_(torch.randn(phi.shape, generator=generator) * torch.logspace(-1, 1.5, phi.shape[0])[:, None])
        model.pos_phi_free.copy_(torch.randn(model.pos_phi_free.shape, generator=generator))
    assert int(bound_phi_frame_condition_(model, KAPPA)) > 0
    tables = (model.prior_bank.phi_embed, model.pos_phi_free)
    snapshot = [table.detach().clone() for table in tables]
    for _ in range(10):
        assert int(bound_phi_frame_condition_(model, KAPPA)) == 0
        assert all(torch.equal(table, old) for table, old in zip(tables, snapshot))


def _windows(n_tokens: int = 4 * 12 + 1) -> DataLoader:
    tokens = torch.arange(n_tokens, dtype=torch.long) % 8
    return DataLoader(TokenWindows(tokens, 4), batch_size=2, shuffle=True, drop_last=True,
                      generator=torch.Generator().manual_seed(5))


def test_bounded_run_resumes_bitwise(tmp_path: Path) -> None:
    cfg = _cfg(phi_scale=1.5, pos_phi_scale=1.0, phi_frame_condition_max=KAPPA, checkpoint_interval=2)
    torch.manual_seed(0)
    straight = VFEModel(cfg)
    vt.train(straight, _windows(), cfg, n_steps=4, log_interval=0, generate_samples=False)

    torch.manual_seed(0)
    model = VFEModel(cfg)
    vt.train(model, _windows(), cfg, n_steps=2, artifacts=RunArtifacts(tmp_path / "run", cfg, model),
             log_interval=0, generate_samples=False)
    resumed = VFEModel(cfg)
    vt.train(resumed, _windows(), cfg, n_steps=4, resume_from=tmp_path / "run" / "checkpoints" / "step_2.pt",
             log_interval=0, generate_samples=False)
    for (name, a), (_, b) in zip(straight.state_dict().items(), resumed.state_dict().items()):
        assert torch.equal(a, b), name


def test_frame_bound_rows_reach_the_counters_and_the_pure_path_flag(tmp_path: Path) -> None:
    cfg = _cfg(phi_scale=1.5, pos_phi_scale=1.0, phi_frame_condition_max=1.5, max_steps=2)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    artifacts = RunArtifacts(tmp_path / "run", cfg, model, save_model_artifacts=False)
    vt.train(model, _windows(), cfg, n_steps=2, artifacts=artifacts, log_interval=1, generate_samples=False)
    rows = [row for row in artifacts.history if "frame_projection_count" in row]
    assert rows and all(math.isfinite(float(row["frame_projection_count"])) for row in rows)
    assert float(rows[-1]["frame_projection_count"]) > 0
    assert float(rows[-1]["frame_projection_total"]) == 2 * (cfg.vocab_size + cfg.max_seq_len)
    assert artifacts.numerics_counters["frame_projection_count"] > 0
    assert _pure_path_report(cfg, artifacts.history)["gauge_flags"]["no_phi_mstep_chart_cap"] is False
    assert _pure_path_report(_cfg(), [])["gauge_flags"]["no_phi_mstep_chart_cap"] is True
