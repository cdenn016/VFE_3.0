r"""Remediation of the 2026-10-31 ultradeep audit, package ``optimizers`` (frame bound, table Fisher steps, weight EMA).

A021 (gap-route-admissibility-fuzz-3): ``pos_phi='frozen'`` with ``pos_phi_project_slk=True`` builds under the frame
condition bound; ``project_phi_to_slk`` keeps a float64 input in float64.
A059 (tables-families-2): the table-step collapse alarm reads the step norm before the schedule factor, so the cosine
tail is not reported as a collapse, while a collapse of the unscheduled norm still is.
A152 (frame-optimizers-2): the bound reports the rows its screen flags and the rows it decomposes.
A153 (frame-optimizers-3): no positional template under ``pos_phi='none'``; the screen is one norm reduction whose
flagged set contains the earlier sum-of-squares set, with every stored value unchanged.
A156 (tables-families-4): the table Fisher step runs on the step's input rows, bitwise the dense step.
A169 (frame-optimizers-1): at ``d >= 8`` a projected row fails the window certificate at its center; under the row-lazy
optimizer an unchanged decomposed row is cleared by its stored copy, bitwise.
A189 (gauge-structure-3): block-local groups contract with the local basis and shared ``block_glk`` reshapes, in the
bound and in the transport.
A201 (tables-families-5): the decode-bank Fisher step reuses the overflow verdict the forward cached.
A246 (frame-optimizers-6): subnormal inputs break the entrywise doubling but not the screen's verdict.
A342 (tables-families-6): the decode-bank Fisher step reports its step health.

Tiny CPU models only (single-digit dimensions, heads and vocabulary).
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest
import torch
from torch.utils.data import DataLoader

import vfe3.gauge_optim as gauge_optim
import vfe3.numerics as numerics
import vfe3.run_artifacts as ra
import vfe3.table_optim as table_optim
import vfe3.train as T
from tests.profiles import live_config, test_config
from vfe3.data.datasets import TokenWindows
from vfe3.gauge_optim import (
    FRAME_BOUND_STAT_KEYS,
    _doubled_traceless_symmetric,
    _frame_bound_rows,
    _head_blocks,
    _HeadBasis,
    _positional_spread,
    _screen_norm_threshold,
    _traceless_symmetric_blocks,
    bound_phi_frame_condition_,
)
from vfe3.geometry.groups import GaugeGroup, get_group
from vfe3.geometry.lie_ops import project_phi_to_slk
from vfe3.model.frame_actions import frame_matvec
from vfe3.model.model import VFEModel
from vfe3.table_optim import (
    DECODE_FISHER_DIAGNOSTIC_KEYS,
    TABLE_FISHER_DIAGNOSTIC_KEYS,
    TableFisherAdamW,
    TableFisherStatistics,
    _overflow_candidate_rows,
    _robust_table_steps,
    _schedule_factor,
    type_counts,
)
from vfe3.train import build_optimizer

CPU = torch.device("cpu")
KAPPA = 20.0


@pytest.fixture(autouse=True)
def _two_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(T, "_save_eval_attention_maps", lambda *args, **kwargs: None)
    monkeypatch.setattr(ra, "_run_figures_isolated", lambda *args, **kwargs: True)
    torch.set_num_threads(2)


def _train_run(cfg: Any, run_dir: Path, *, n_steps: int, log_interval: int = 1) -> Tuple[List[float], Any]:
    V, N = cfg.vocab_size, cfg.max_seq_len
    torch.manual_seed(5)
    model = VFEModel(cfg)
    artifacts = ra.RunArtifacts(run_dir, cfg, model, save_model_artifacts=False)
    train = TokenWindows((torch.arange(61) * 5 + 2) % V, N)
    val = TokenWindows((torch.arange(23) * 3) % V, N, pad_final=True)
    loaders = (DataLoader(train, batch_size=cfg.batch_size, shuffle=False, drop_last=True),
               DataLoader(val, batch_size=2, shuffle=False))
    torch.manual_seed(11)
    losses = T.train(model, loaders[0], cfg, n_steps=n_steps, grad_clip=cfg.grad_clip, log_interval=log_interval,
                     eval_interval=0, val_loader=loaders[1], device=CPU, logger=logging.getLogger("audit-1031"),
                     artifacts=artifacts, generate_samples=False)
    return losses, artifacts


def _warning_kinds(run_dir: Path) -> List[str]:
    path = run_dir / T.RUN_WARNINGS_LOG
    if not path.exists():
        return []
    return [json.loads(line)["kind"] for line in path.read_text(encoding="utf-8").splitlines()]


def _metrics_rows(run_dir: Path) -> List[Dict[str, str]]:
    import csv
    with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# ------------------------------------------------------------------------------------------------ A021

def test_frozen_projected_positional_factor_builds_under_the_frame_condition_bound() -> None:
    r"""The live tiny route with ``pos_phi='frozen'`` and ``pos_phi_project_slk=True`` (admitted by validation) raised
    ``expected m1 and m2 to have the same dtype`` at construction, where the bound measures the positional spread from
    float64 coordinates."""
    cfg = live_config(pos_phi="frozen", pos_phi_project_slk=True, phi_frame_condition_max=20.0)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    tokens = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 1]])
    logits = model(tokens)[0]
    assert bool(torch.isfinite(logits).all())


def test_project_phi_to_slk_keeps_a_float64_input_in_float64() -> None:
    group = get_group("block_glk")(8, 2)                                    # float32 elementary basis, d = 4
    generator = torch.Generator().manual_seed(1)
    phi64 = torch.randn(5, group.generator_shape[0], generator=generator, dtype=torch.float64)
    out64 = project_phi_to_slk(phi64, group, group.irrep_dims)
    assert out64.dtype == torch.float64
    traces = out64.reshape(5, 2, 4, 4).diagonal(dim1=-2, dim2=-1).sum(-1)  # (5, H) per-block traces
    assert float(traces.abs().max()) < 1e-14
    out32 = project_phi_to_slk(phi64.to(torch.float32), group, group.irrep_dims)
    assert out32.dtype == torch.float32
    torch.testing.assert_close(out64.to(torch.float32), out32, rtol=0.0, atol=1e-6)


# ------------------------------------------------------------------------------------------------ A201

def _decode_fisher_cfg(**overrides: Any) -> Any:
    values: Dict[str, Any] = dict(vocab_size=9, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=1,
                                  decode_mode="diagonal_chunked", untie_decode_bank=True,
                                  m_decode_update_mode="fisher", m_table_update_mode="fisher",
                                  m_table_fisher_lr=0.5, m_table_fisher_trust_radius=0.05)
    values.update(overrides)
    return test_config(**values)


def test_decode_fisher_step_reuses_the_cached_overflow_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""After a forward and backward, ``optimizer.step()`` ran one overflow reduction (and one immediate host read) per
    step for the decode variances, because ``decode_variances`` detached the table before the cached check."""
    cfg = _decode_fisher_cfg()
    torch.manual_seed(0)
    model = VFEModel(cfg)
    optimizer = build_optimizer(model, cfg)
    tokens = torch.tensor([[1, 2, 3, 4], [4, 5, 1, 6]])
    targets = torch.tensor([[2, 3, 4, 5], [5, 1, 6, 7]])
    reductions: List[int] = []
    original = numerics._max_log_exceeded

    def counting(log_sigma: torch.Tensor, max_log: float) -> torch.Tensor:
        reductions.append(int(log_sigma.shape[0]))
        return original(log_sigma, max_log)

    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        _, loss, _ = model(tokens, targets)
        loss.backward()
        optimizer.table_statistics_provider.set_batch(tokens, torch.tensor(targets.numel()), targets=targets)
        before = model.prior_bank.decode_mu_embed.detach().clone()
        with monkeypatch.context() as m:
            m.setattr(numerics, "_max_log_exceeded", counting)
            reductions.clear()
            optimizer.step()
        assert reductions == []                                             # both variance reads hit the cache
        assert not torch.equal(model.prior_bank.decode_mu_embed.detach(), before)   # the decode step still ran


def test_decode_variances_equal_the_bounded_table() -> None:
    cfg = _decode_fisher_cfg()
    model = VFEModel(cfg)
    optimizer = build_optimizer(model, cfg)
    pb = model.prior_bank
    t = optimizer.table_statistics_provider.decode_variances(torch.float32)
    assert not t.requires_grad
    expected = numerics.bounded_variance_from_log(pb.decode_sigma_log_embed.detach(), eps=pb.eps)
    assert torch.equal(t, expected)


# ------------------------------------------------------------------------------------------------ A342

def test_decode_fisher_step_reports_its_step_health() -> None:
    r"""The decode diagnostics equal an independent float64 evaluation of the module-docstring step
    :math:`\Delta_v = -\eta\,N/\max(c_v, 1)\,(t_v\odot g_m, 2 g_l)`, scaled to Fisher norm at most :math:`r`."""
    cfg = _decode_fisher_cfg(decode_mu_weight_decay=0.0, sigma_weight_decay=0.0, mu_weight_decay=0.0)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    optimizer = build_optimizer(model, cfg)
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
        group["lr"] = 0.5 * group["lr"]                                     # schedule factor 1/2
    tokens = torch.tensor([[1, 2, 3, 4], [4, 5, 1, 6]])
    targets = torch.tensor([[2, 3, 4, 5], [5, 1, 6, -100]])
    _, loss, _ = model(tokens, targets)
    loss.backward()
    pb = model.prior_bank
    g_m = pb.decode_mu_embed.grad.detach().double().clone()
    g_l = pb.decode_sigma_log_embed.grad.detach().double().clone()
    t = numerics.bounded_variance_from_log(pb.decode_sigma_log_embed.detach(), eps=pb.eps).double()
    optimizer.table_statistics_provider.set_batch(tokens, torch.tensor(7), targets=targets)
    optimizer._collect_table_diag = True
    optimizer.step()
    diag = optimizer._decode_diag
    assert set(diag) == set(DECODE_FISHER_DIAGNOSTIC_KEYS)
    counts = torch.zeros(cfg.vocab_size, dtype=torch.float64).index_add_(
        0, targets[targets >= 0], torch.ones(int((targets >= 0).sum()), dtype=torch.float64))
    weight = (7.0 / counts.clamp_min(1.0))[:, None]
    d_m, d_l = -cfg.m_table_fisher_lr * weight * t * g_m, -cfg.m_table_fisher_lr * weight * 2.0 * g_l
    norm = ((d_m.square() / t).sum(-1) + 0.5 * d_l.square().sum(-1)).sqrt()    # (V,) ||d||
    moved = norm > 0
    trust = (cfg.m_table_fisher_trust_radius / norm[moved]).clamp(max=1.0)
    unscheduled = trust * norm[moved]                                        # min(||d||, r)
    assert diag["decode_fisher_active_rows"] == float(moved.sum()) > 0
    assert diag["decode_fisher_trust_scale_mean"] == pytest.approx(float(trust.mean()), rel=1e-6)
    assert diag["decode_fisher_trust_scale_min"] == pytest.approx(float(trust.min()), rel=1e-6)
    assert diag["decode_fisher_unscheduled_step_norm_mean"] == pytest.approx(float(unscheduled.mean()), rel=1e-6)
    assert diag["decode_fisher_step_norm_mean"] == pytest.approx(0.5 * float(unscheduled.mean()), rel=1e-6)
    assert float(trust.min()) < 1.0                                          # the radius binds on some row


def test_decode_fisher_columns_reach_the_metrics_rows(tmp_path: Path) -> None:
    cfg = _decode_fisher_cfg(vocab_size=7, warmup_steps=0, max_steps=2, log_interval=1, eval_interval=0,
                             checkpoint_interval=0, batch_size=2)
    _train_run(cfg, tmp_path / "run", n_steps=2)
    rows = _metrics_rows(tmp_path / "run")
    assert rows and all(key in rows[0] for key in DECODE_FISHER_DIAGNOSTIC_KEYS)
    assert float(rows[-1]["decode_fisher_active_rows"]) > 0
    off = _decode_fisher_cfg(vocab_size=7, warmup_steps=0, max_steps=1, log_interval=1, eval_interval=0,
                             checkpoint_interval=0, batch_size=2, m_decode_update_mode="adamw")
    _train_run(off, tmp_path / "off", n_steps=1)
    assert not any(key in _metrics_rows(tmp_path / "off")[0] for key in DECODE_FISHER_DIAGNOSTIC_KEYS)


# ------------------------------------------------------------------------------------------------ A156

@torch.no_grad()
def _dense_table_fisher_step(self: Any) -> None:
    r"""The table Fisher step as it stood before 2026-10-31 (vfe3/table_optim.py at 5bb0a57): every quantity on all
    (V, K) rows. Kept here verbatim as the parity reference; it reports the four diagnostics it had."""
    mu_group, sigma_group = self._table_fisher_groups()
    mu = mu_group["params"][0] if mu_group is not None else None
    log_s = sigma_group["params"][0] if sigma_group is not None else None
    live = [p for p in (mu, log_s) if p is not None and p.grad is not None]
    if self._collect_table_diag:
        self._table_diag = {}
    if not live:
        return
    grad_scale, found_inf = getattr(self, "grad_scale", None), getattr(self, "found_inf", None)
    tokens, scored = self.table_statistics_provider.batch()
    reference = live[0]
    dtype = torch.promote_types(reference.dtype, torch.float32)
    V = reference.shape[0]
    skip = None if found_inf is None else found_inf.to(device=reference.device) != 0
    count = type_counts(tokens, V, reference.device).to(dtype)
    total = scored.to(device=reference.device, dtype=dtype).clamp_min(1.0)
    weight = (total / count.clamp_min(1.0)).unsqueeze(-1)
    s = self.table_statistics_provider.variances(dtype)

    def gradient(p: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if p is None or p.grad is None:
            return None
        g = p.grad.to(dtype)
        return g if grad_scale is None else g / grad_scale.to(device=g.device, dtype=dtype)

    g_mu, g_l = gradient(mu), gradient(log_s)
    frames = self.table_statistics_provider.frames() if g_mu is not None else None
    if frames is not None:
        rows, frame = frames
        rows = rows.to(device=g_mu.device)
        forward = frame.forward.to(device=g_mu.device)
        g_mu = g_mu.index_copy(0, rows, frame_matvec(forward.transpose(-1, -2), g_mu[rows], frame.block_diagonal))
    eta = self._table_fisher_lr
    step_mu = None if g_mu is None else -eta * weight * s * g_mu
    step_l = None if g_l is None else -eta * weight * 2.0 * g_l
    norm_sq_mu = None if step_mu is None else (step_mu.square() / s).sum(-1)
    norm_sq_l = None if step_l is None else 0.5 * step_l.square().sum(-1)
    norm_sq = torch.zeros(V, dtype=dtype, device=reference.device)
    if norm_sq_mu is not None:
        norm_sq = norm_sq + norm_sq_mu
    if norm_sq_l is not None:
        norm_sq = norm_sq + norm_sq_l
    norm = norm_sq.sqrt()
    active = count > 0
    trust = (self._table_fisher_trust_radius / norm.clamp_min(torch.finfo(dtype).tiny)).clamp(max=1.0)
    trust = torch.where(active, trust, torch.zeros_like(trust)).unsqueeze(-1)
    overflow = active & ~torch.isfinite(norm_sq)
    cand = _overflow_candidate_rows(tokens, V, reference.device)
    cand_overflow = overflow[cand]
    wide_mu, wide_l, wide_sq_mu, wide_sq_l, wide_trust = _robust_table_steps(
        None if g_mu is None else g_mu[cand], None if g_l is None else g_l[cand], weight[cand], s[cand], eta,
        self._table_fisher_trust_radius, active[cand])
    realized_sq = torch.zeros(V, dtype=dtype, device=reference.device)
    overflow_sq = torch.zeros(V, dtype=torch.float64, device=reference.device)
    for p, step, group, part, wide, wide_sq in ((mu, step_mu, mu_group, norm_sq_mu, wide_mu, wide_sq_mu),
                                                (log_s, step_l, sigma_group, norm_sq_l, wide_l, wide_sq_l)):
        if step is None:
            continue
        decay = float(group["lr"]) * float(group.get("weight_decay", 0.0))
        if decay != 0.0:
            if skip is None:
                p.mul_(1.0 - decay)
            else:
                keep = torch.full((), 1.0 - decay, dtype=p.dtype, device=p.device)
                p.mul_(torch.where(skip, torch.ones_like(keep), keep))
        factor = _schedule_factor(group)
        if factor != 0.0:
            scheduled = torch.where(trust > 0, factor * trust * step, torch.zeros_like(step))
            scheduled = scheduled.index_put(
                (cand,), torch.where(cand_overflow.unsqueeze(-1), (factor * wide).to(dtype), scheduled[cand]))
            if frames is not None and p is mu:
                scheduled = scheduled.index_copy(0, rows, frame_matvec(forward, scheduled[rows], frame.block_diagonal))
            update = scheduled.to(p.dtype)
            p.add_(update if skip is None else torch.where(skip, torch.zeros_like(update), update))
            realized_sq = realized_sq + factor ** 2 * torch.where(overflow, torch.zeros_like(part), part)
            overflow_sq = overflow_sq.index_put((cand,), overflow_sq[cand] + factor ** 2 * wide_sq)
        p.grad = None
    if self._collect_table_diag:
        accepted = trust.squeeze(-1).index_put(
            (cand,), torch.where(cand_overflow, wide_trust.to(dtype), trust.squeeze(-1)[cand]))
        n_active = active.sum().to(dtype)
        realized = torch.where(overflow, overflow_sq.sqrt().to(dtype), accepted * realized_sq.sqrt())
        values = torch.stack([
            n_active,
            torch.where(active, accepted, torch.zeros_like(accepted)).sum() / n_active,
            torch.where(active, accepted, torch.full_like(accepted, float("inf"))).amin(),
            torch.where(active, realized, torch.zeros_like(realized)).sum() / n_active,
            (skip if skip is not None else torch.zeros((), dtype=torch.bool, device=reference.device)).to(dtype),
        ]).cpu().tolist()
        if values[4]:
            return
        nan = float("nan")
        self._table_diag = dict(zip(TABLE_FISHER_DIAGNOSTIC_KEYS[:4], (
            values[0], *((nan, nan, nan) if values[0] == 0.0 else values[1:4]))))


def _same_diag(new: Dict[str, float], old: Dict[str, float]) -> None:
    assert set(old) <= set(new)
    for key, value in old.items():
        assert (math.isnan(value) and math.isnan(new[key])) or new[key] == value, key


class _Bank(torch.nn.Module):
    def __init__(self, log_s: torch.Tensor) -> None:
        super().__init__()
        self.log_s, self.eps = log_s, 1e-12

    def _prior_sigma_log_table(self) -> torch.Tensor:
        return self.log_s


@pytest.mark.parametrize("found_inf", [None, 0.0, 1.0])
@pytest.mark.parametrize("schedule", [1.0, 0.37])
@pytest.mark.parametrize("tokens", [
    torch.tensor([[0, 1, 3, 3], [5, 7, 1, 0]]),                            # 8 ids < V: rows 2, 4, 6, 8-10 absent
    torch.tensor([[2, 2, 0, 1, 4, 9], [3, 5, 6, 7, 8, 10]]),                # 12 ids >= V = 11: every row
])
def test_row_restricted_table_step_is_bitwise_the_dense_step(found_inf: Optional[float], schedule: float,
                                                             tokens: torch.Tensor) -> None:
    V, K, radius = 11, 4, 0.5
    gen = torch.Generator().manual_seed(0)
    mu0 = torch.randn(V, K, generator=gen)
    l0 = 0.3 * torch.randn(V, K, generator=gen)
    log_s = torch.log(torch.rand(V, K, generator=gen) * 4.0 + 1.0)
    g_mu = torch.randn(V, K, generator=gen) * 1e-2
    g_l = torch.randn(V, K, generator=gen) * 1e-2
    g_mu[0], g_mu[3] = 1e17, -1e35                                          # norm_sq overflow, step overflow
    g_l[5] = 2e30
    g_mu[2] = 0.7                                                           # a gradient on a row absent from the step
    g_mu[1] *= 300.0                                                        # a trust-binding row

    def run(step: Callable[[Any], None]) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        mu, log_v = torch.nn.Parameter(mu0.clone()), torch.nn.Parameter(l0.clone())
        optimizer = TableFisherAdamW(
            [{"params": [mu], "lr": schedule, "initial_lr": 1.0, "weight_decay": 0.01, "table_fisher": "mu"},
             {"params": [log_v], "lr": schedule, "initial_lr": 1.0, "weight_decay": 0.0, "table_fisher": "sigma"}],
            TableFisherStatistics(_Bank(log_s)), table_fisher_lr=1.0, table_fisher_trust_radius=radius)
        if found_inf is not None:
            optimizer.found_inf = torch.tensor(found_inf)
        optimizer._collect_table_diag = True
        mu.grad, log_v.grad = g_mu.clone(), g_l.clone()
        optimizer.table_statistics_provider.set_batch(tokens, torch.tensor(8192))
        step(optimizer)
        return {"mu": mu.detach().clone(), "log_s": log_v.detach().clone()}, dict(optimizer._table_diag)

    new, new_diag = run(lambda optimizer: optimizer._table_fisher_step())
    old, old_diag = run(_dense_table_fisher_step)
    for name in new:
        assert torch.equal(new[name], old[name]), name
    _same_diag(new_diag, old_diag)
    if found_inf != 1.0:
        assert bool(torch.isfinite(new["mu"]).all()) and new_diag["table_fisher_active_rows"] > 0


def test_row_restricted_frame_diagonal_step_is_bitwise_the_dense_step(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_frame_diagonal_table_fisher_20261018 import _cfg as _frame_cfg, _model as _frame_model
    cfg = _frame_cfg(m_table_fisher_lr=0.3, m_table_fisher_trust_radius=0.5)
    reference = _frame_model(cfg).float()
    tokens = torch.tensor([[1, 2, 2], [1, 3, 7]])                           # 6 ids < V = 9, type 2 and 1 repeated
    gen = torch.Generator().manual_seed(3)
    grads = {name: torch.randn(p.shape, generator=gen) * 1e-2 for name, p in reference.named_parameters()
             if p.requires_grad}
    grads["prior_bank.mu_embed"][2] = 1e20                                  # overflow on a repeated type
    grads["prior_bank.sigma_log_embed"][7] = -1e25

    def run() -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        model = _frame_model(cfg).float()
        optimizer = build_optimizer(model, cfg)
        assert optimizer.table_statistics_provider.reads_frames
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.grad = grads[name].clone()
        optimizer._collect_table_diag = True
        optimizer.table_statistics_provider.set_batch(tokens, torch.tensor(6))
        optimizer.step()
        return ({name: p.detach().clone() for name, p in model.named_parameters()}, dict(optimizer._table_diag))

    new, new_diag = run()
    with monkeypatch.context() as m:
        m.setattr(table_optim._TableFisherMixin, "_table_fisher_step", _dense_table_fisher_step)
        old, old_diag = run()
    for name in new:
        assert torch.equal(new[name], old[name]), name
    _same_diag(new_diag, old_diag)


def test_row_restricted_step_is_bitwise_on_the_tiny_live_route(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""Three accepted ``train_step`` calls of the tiny live route: the loss, every parameter gradient the optimizer
    receives, every parameter after each step and the table diagnostics equal those of the dense step."""
    cfg = live_config()
    assert cfg.m_table_update_mode == "fisher"
    g = torch.Generator().manual_seed(4)
    batches = [(torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len), generator=g),
                torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len), generator=g)) for _ in range(3)]

    def run() -> List[Tuple[Any, ...]]:
        torch.manual_seed(0)
        model = VFEModel(cfg)
        optimizer = build_optimizer(model, cfg)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: (1.0, 0.5, 0.25)[min(step, 2)])
        names = {id(p): n for n, p in model.named_parameters()}
        seen: Dict[str, torch.Tensor] = {}
        optimizer.register_step_pre_hook(lambda *_: seen.update(
            {names[id(p)]: p.grad.clone() for group in optimizer.param_groups for p in group["params"]
             if p.grad is not None}))
        record = []
        torch.manual_seed(1)
        for tokens, targets in batches:
            seen.clear()
            optimizer._collect_table_diag = True
            loss = T.train_step(model, optimizer, scheduler, tokens, targets)
            record.append((loss, dict(seen), {n: p.detach().clone() for n, p in model.named_parameters()},
                           dict(optimizer._table_diag)))
        return record

    new = run()
    with monkeypatch.context() as m:
        m.setattr(table_optim._TableFisherMixin, "_table_fisher_step", _dense_table_fisher_step)
        old = run()
    for (loss_n, grads_n, params_n, diag_n), (loss_o, grads_o, params_o, diag_o) in zip(new, old):
        assert loss_n == loss_o
        assert grads_n.keys() == grads_o.keys() and grads_n
        for name in grads_n:
            assert torch.equal(grads_n[name], grads_o[name]), name
        for name in params_n:
            assert torch.equal(params_n[name], params_o[name]), name
        _same_diag(diag_n, diag_o)
    assert new[-1][3]["table_fisher_active_rows"] > 0


def test_table_variances_read_on_rows_equal_the_full_table_rows() -> None:
    gen = torch.Generator().manual_seed(2)
    log_s = torch.randn(9, 4, generator=gen) * 3.0
    log_s[4, 1] = 95.0                                                      # above max_log: clamped before exp
    statistics = TableFisherStatistics(_Bank(log_s))
    rows = torch.tensor([3, 4, 4, 0, 8])
    with pytest.warns(RuntimeWarning):
        full = statistics.variances(torch.float32)
    with pytest.warns(RuntimeWarning):
        gathered = statistics.variances(torch.float32, index=rows)
    assert torch.equal(gathered, full[rows])


# ------------------------------------------------------------------------------------------------ A059

def _table_step_cfg(**overrides: Any) -> Any:
    values: Dict[str, Any] = dict(vocab_size=7, embed_dim=4, n_heads=2, max_seq_len=4, batch_size=2, n_layers=1,
                                  decode_mode="linear", m_table_update_mode="fisher", m_table_fisher_lr=50.0,
                                  m_table_fisher_trust_radius=0.1, warmup_steps=0, max_steps=3, min_lr_frac=0.0,
                                  log_interval=1, eval_interval=0, checkpoint_interval=0)
    values.update(overrides)
    return test_config(**values)


@pytest.mark.parametrize("key, fires", [(T.TABLE_STEP_COLLAPSE_KEY, False), ("table_fisher_step_norm_mean", True)])
def test_the_schedule_tail_is_not_reported_as_a_table_collapse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                              key: str, fires: bool) -> None:
    r"""A schedule multiplier falling 1e6-fold between two log rows (as the live cosine to zero does between steps
    29,000 and 30,000) moves the scheduled norm, which the alarm read until 2026-10-31, and not the unscheduled one."""
    multipliers = (1.0, 1.0, 1e-6)
    monkeypatch.setattr(T, "lr_lambda", lambda step, cfg: multipliers[min(int(step), 2)])
    monkeypatch.setattr(T, "TABLE_STEP_COLLAPSE_KEY", key)
    _train_run(_table_step_cfg(), tmp_path / "run", n_steps=3)
    rows = _metrics_rows(tmp_path / "run")
    scheduled = [float(row["table_fisher_step_norm_mean"]) for row in rows]
    unscheduled = [float(row["table_fisher_unscheduled_step_norm_mean"]) for row in rows]
    assert scheduled[2] * 1e4 < scheduled[1]                                # the schedule factor alone
    assert max(unscheduled) < 10.0 * min(unscheduled)
    assert ("table_step_collapse" in _warning_kinds(tmp_path / "run")) is fires


def test_a_collapse_of_the_unscheduled_table_step_still_warns(caplog: pytest.LogCaptureFixture) -> None:
    r"""The real diagnostic on a cosine-to-zero tail with fixed gradients, and with gradients falling 1e4-fold."""
    V, K, radius = 9, 4, 0.5
    gen = torch.Generator().manual_seed(1)
    log_s = torch.log(torch.rand(V, K, generator=gen) + 0.5)
    g_mu, g_l = torch.randn(V, K, generator=gen), torch.randn(V, K, generator=gen)
    tokens = torch.tensor([[0, 1, 2, 3, 3, 5]])
    cfg = _table_step_cfg(max_steps=30000, warmup_steps=250)

    def diagnostic(factor: float, gradient_scale: float) -> Dict[str, float]:
        mu, log_v = torch.nn.Parameter(torch.zeros(V, K)), torch.nn.Parameter(torch.zeros(V, K))
        optimizer = TableFisherAdamW(
            [{"params": [mu], "lr": factor, "initial_lr": 1.0, "weight_decay": 0.0, "table_fisher": "mu"},
             {"params": [log_v], "lr": factor, "initial_lr": 1.0, "weight_decay": 0.0, "table_fisher": "sigma"}],
            TableFisherStatistics(_Bank(log_s)), table_fisher_lr=0.125, table_fisher_trust_radius=radius)
        optimizer._collect_table_diag = True
        mu.grad, log_v.grad = gradient_scale * g_mu, gradient_scale * g_l
        optimizer.table_statistics_provider.set_batch(tokens, torch.tensor(6))
        optimizer._table_fisher_step()
        return optimizer._table_diag

    logger = logging.getLogger("audit-1031-alarm")
    tail = [diagnostic(T.lr_lambda(step, cfg), 1.0) for step in (28999, 29999)]
    assert tail[1]["table_fisher_step_norm_mean"] * 1e5 < tail[0]["table_fisher_step_norm_mean"]
    assert not T._warn_table_step_collapse(tail[0][T.TABLE_STEP_COLLAPSE_KEY], tail[1][T.TABLE_STEP_COLLAPSE_KEY],
                                           logger, step=30000)
    healthy, collapsed = diagnostic(0.5, 1e-1), diagnostic(0.5, 1e-5)
    with caplog.at_level(logging.WARNING, logger="audit-1031-alarm"):
        assert T._warn_table_step_collapse(healthy[T.TABLE_STEP_COLLAPSE_KEY], collapsed[T.TABLE_STEP_COLLAPSE_KEY],
                                           logger, step=1000)
    assert T.TABLE_STEP_COLLAPSE_KEY in caplog.text
    for diag in (*tail, healthy, collapsed):
        assert diag[T.TABLE_STEP_COLLAPSE_KEY] <= radius * (1 + 1e-6)


# ------------------------------------------------------------------------------------------------ frame bound helpers

def _bound_model(**overrides: Any) -> VFEModel:
    values: Dict[str, Any] = dict(vocab_size=9, embed_dim=8, n_heads=2, max_seq_len=3, gauge_group="block_glk",
                                  pos_phi="none", phi_frame_condition_max=KAPPA)
    values.update(overrides)
    torch.manual_seed(0)
    return VFEModel(test_config(**values))


class _FrameTables(torch.nn.Module):
    r"""The attributes the bound reads, without a model: a two-head ``block_glk`` group of width ``d``, a trainable
    token frame table of ``rows`` rows and the two config fields it consults (no positional factor)."""

    def __init__(self, d: int, rows: int = 7, *, sparse_rows: bool = False) -> None:
        super().__init__()
        from types import SimpleNamespace
        self.group = get_group("block_glk")(2 * d, 2)
        self.prior_bank = torch.nn.Module()
        self.prior_bank.phi_embed = torch.nn.Parameter(torch.zeros(rows, self.group.generator_shape[0]))
        self.cfg = SimpleNamespace(m_phi_sparse_rows=sparse_rows, pos_phi="none")


def _rows_with_spectrum(lam: torch.Tensor, n_rows: int, H: int, generator: torch.Generator) -> torch.Tensor:
    r"""(n_rows, H d d) float64 row-major blocks with traceless symmetric spectrum ``lam``, a skew part and a trace."""
    d = lam.shape[-1]
    Q, _ = torch.linalg.qr(torch.randn(n_rows, H, d, d, generator=generator, dtype=torch.float64))
    S0 = Q @ torch.diag_embed(lam.to(torch.float64).expand(n_rows, H, d)) @ Q.mT
    skew = torch.randn(n_rows, H, d, d, generator=generator, dtype=torch.float64)
    X = 0.5 * (S0 + S0.mT) + 0.3 * (skew - skew.mT) + 0.2 * torch.eye(d, dtype=torch.float64)
    return X.reshape(n_rows, -1)


def _exact_spread(rows: torch.Tensor, H: int, d: int) -> torch.Tensor:          # (r, H) float64
    X = rows.detach().to(torch.float64).reshape(-1, H, d, d)
    S = 0.5 * (X + X.mT)
    S0 = S - S.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None] * torch.eye(d, dtype=torch.float64)
    lam = torch.linalg.eigvalsh(S0)
    return lam[..., -1] - lam[..., 0]


def _counting_eigh(monkeypatch: pytest.MonkeyPatch) -> List[int]:
    eigh, blocks = torch.linalg.eigh, []

    def counting(A: torch.Tensor) -> tuple:
        blocks.append(A.numel() // (A.shape[-1] * A.shape[-1]))
        return eigh(A)

    monkeypatch.setattr(torch.linalg, "eigh", counting)
    return blocks


# ------------------------------------------------------------------------------------------------ A152

def test_bound_reports_flagged_and_decomposed_rows_without_a_host_read() -> None:
    model = _bound_model()
    H, d, L = 2, 4, math.log(KAPPA)
    a = 0.4 * L                                                             # spectrum (a, a, -a, -a): spread 0.8 L
    assert math.sqrt(2.0) * 2.0 * a > L                                     # yet sqrt(2) ||S_0||_F > 2c = L: flagged
    generator = torch.Generator().manual_seed(0)
    inside = _rows_with_spectrum(torch.tensor([-a, -a, a, a]), 4, H, generator)
    outside = _rows_with_spectrum(torch.tensor([-2.0, -1.0, 1.0, 2.0]), 3, H, generator)   # spread 4 > L
    with torch.no_grad():
        phi = model.prior_bank.phi_embed
        phi.copy_(1e-3 * torch.randn(phi.shape, generator=generator))
        phi[:4].copy_(inside)
        phi[4:7].copy_(outside)
    before = phi.detach().clone()
    stats: Dict[str, float] = {}
    changed = bound_phi_frame_condition_(model, KAPPA, stats_out=stats)
    assert stats == {"frame_bound_flagged_rows": 7, "frame_bound_decomposed_rows": 3, "frame_bound_float64_rows": 0}
    assert int(changed) == 3
    assert torch.equal(phi.detach()[:4], before[:4]) and torch.equal(phi.detach()[7:], before[7:])
    stats.clear()
    bound_phi_frame_condition_(model, KAPPA, stats_out=stats)               # nothing over the bound any more
    assert stats["frame_bound_flagged_rows"] == 7 and set(stats) == set(FRAME_BOUND_STAT_KEYS)


def test_train_step_writes_the_bound_counts_on_metrics_steps_only() -> None:
    cfg = test_config(vocab_size=9, embed_dim=8, n_heads=2, max_seq_len=3, gauge_group="block_glk", pos_phi="none",
                      phi_frame_condition_max=1.5, phi_scale=1.0)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    tokens, targets = torch.tensor([[0, 1, 2], [3, 4, 5]]), torch.tensor([[1, 2, 3], [4, 5, 6]])
    metrics: Dict[str, float] = {}
    T.train_step(model, optimizer, scheduler, tokens, targets, metrics_out=metrics)
    assert set(FRAME_BOUND_STAT_KEYS) <= set(metrics)
    assert metrics["frame_bound_flagged_rows"] >= metrics["frame_bound_decomposed_rows"] >= 0


# ------------------------------------------------------------------------------------------------ A153

def test_no_positional_template_under_pos_phi_none(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _bound_model()
    called: List[int] = []
    monkeypatch.setattr(model, "_pos_phi_right", lambda phi: called.append(1))
    assert _positional_spread(model, None, 2, 4) is None
    assert called == []


@pytest.mark.parametrize("pos_phi, project", [("learned", False), ("frozen", False), ("frozen", True)])
def test_positional_spread_is_unchanged_by_the_stride_zero_template(pos_phi: str, project: bool) -> None:
    model = _bound_model(pos_phi=pos_phi, pos_phi_project_slk=project, pos_phi_scale=0.3, max_seq_len=4)
    group = model.group
    template = torch.zeros(model.cfg.max_seq_len, group.generator_shape[0], dtype=torch.float64)
    Y = model._pos_phi_right(template)
    lam = torch.linalg.eigvalsh(_traceless_symmetric_blocks(Y.detach(), None, 2, 4, torch.float64))
    expected = (lam[..., -1] - lam[..., 0]).amax(dim=0)
    assert torch.equal(_positional_spread(model, None, 2, 4), expected)


def _old_screen(T2: torch.Tensor, screen: torch.Tensor) -> torch.Tensor:          # the pre-2026-10-31 test
    return (T2.square().sum(dim=(-2, -1)) > 4.0 * screen).any(-1)


def _new_screen(T2: torch.Tensor, screen: torch.Tensor) -> torch.Tensor:
    return gauge_optim._screen_flags(T2, _screen_norm_threshold(screen, T2.shape[-1] ** 2))


@pytest.mark.parametrize("d", [3, 8, 32])
def test_norm_screen_flags_a_superset_and_stays_sound(d: int) -> None:
    H, c, u = 2, 0.5 * math.log(KAPPA), 2.0 ** -24
    screen = torch.full((H,), 0.5 * (2.0 * c) ** 2 * (1.0 - 1e-4), dtype=torch.float64).to(torch.float32)
    gen = torch.Generator().manual_seed(d)
    cases = []
    for scale, trace in ((1.0, 0.0), (0.3, 0.0), (5.0, 40.0), (1e-3, 0.0), (1e19, 0.0), (1e-39, 0.0)):
        phi = scale * torch.randn(300, H * d * d, generator=gen)
        phi.view(300, H, d, d).diagonal(dim1=-2, dim2=-1).add_(trace)
        cases.append(phi)
    for phi in cases:
        T2 = _doubled_traceless_symmetric(phi, None, H, d, torch.float32)
        old, new = _old_screen(T2, screen), _new_screen(T2, screen)
        assert not bool((old & ~new).any())                                     # every old flag is a new flag
    # Blocks on the threshold: T scaled so that ||T||_F sits within a few band widths of sqrt(4 screen), both sides.
    gamma = d * d * u / (1.0 - d * d * u)
    base = _doubled_traceless_symmetric(torch.randn(400, H * d * d, generator=gen), None, H, d, torch.float32)
    target = (4.0 * screen.double()).sqrt()                                     # (H,)
    norms = torch.linalg.vector_norm(base.double(), dim=(-2, -1))               # (400, H)
    jitter = 1.0 + (4.0 * gamma + 2e-5) * torch.linspace(-1, 1, 400, dtype=torch.float64)[:, None]
    boundary = (base.double() * (target / norms * jitter)[..., None, None]).to(torch.float32)
    old, new = _old_screen(boundary, screen), _new_screen(boundary, screen)
    assert bool(old.any()) and bool((~new).any()) and not bool((old & ~new).any())
    ratio = (torch.linalg.vector_norm(boundary.double(), dim=(-2, -1)) / target).amax(-1)   # (400,)
    assert bool((ratio[new & ~old] >= 1.0 - 2.0 * gamma - 8.0 * u).all())          # the extra flags: a thin band
    # Soundness: an unflagged block has spread <= sqrt(2) ||S_0||_F <= 2c in exact arithmetic of the computed block.
    exact = torch.linalg.eigvalsh(boundary.double() / 2.0)
    spread = (exact[..., -1] - exact[..., 0])[~new]
    assert bool((spread <= 2.0 * c).all())


def _stretched_tables(d: int, seed: int, **kwargs: Any) -> _FrameTables:
    model = _FrameTables(d, **kwargs)
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        phi = model.prior_bank.phi_embed
        phi.copy_(torch.randn(phi.shape, generator=gen) * torch.logspace(-1.5, 0.3, phi.shape[0])[:, None])
    return model


@pytest.mark.parametrize("make", [
    lambda: _stretched_tables(4, 7),
    lambda: _stretched_tables(8, 7),
    lambda: _bound_model(pos_phi="learned"),
])
def test_bound_with_the_norm_screen_stores_the_values_of_the_sum_of_squares_screen(
        make: Callable[[], torch.nn.Module], monkeypatch: pytest.MonkeyPatch) -> None:
    def tables(model: torch.nn.Module) -> List[torch.Tensor]:
        out = [model.prior_bank.phi_embed.detach().clone()]
        if getattr(model, "pos_phi_free", None) is not None:
            out.append(model.pos_phi_free.detach().clone())
        return out

    new_model, old_model = make(), make()
    for model in (new_model, old_model):
        gen = torch.Generator().manual_seed(8)
        with torch.no_grad():
            phi = model.prior_bank.phi_embed
            phi.copy_(torch.randn(phi.shape, generator=gen) * torch.logspace(-1.5, 0.3, phi.shape[0])[:, None])
            if getattr(model, "pos_phi_free", None) is not None:
                model.pos_phi_free.copy_(0.5 * torch.randn(model.pos_phi_free.shape, generator=gen))
    flagged_new: List[int] = []
    for _ in range(3):
        stats: Dict[str, float] = {}
        bound_phi_frame_condition_(new_model, KAPPA, stats_out=stats)
        flagged_new.append(stats["frame_bound_flagged_rows"])
        with monkeypatch.context() as m:                                    # the sum-of-squares screen
            m.setattr(gauge_optim, "_screen_norm_threshold", lambda screen, n: 4.0 * screen)
            m.setattr(gauge_optim, "_screen_flags",
                      lambda T2, threshold: (T2.square().sum(dim=(-2, -1)) > threshold).any(-1))
            old_stats: Dict[str, float] = {}
            bound_phi_frame_condition_(old_model, KAPPA, stats_out=old_stats)
        assert stats["frame_bound_flagged_rows"] >= old_stats["frame_bound_flagged_rows"]
        for a, b in zip(tables(new_model), tables(old_model)):
            assert torch.equal(a, b)
        with torch.no_grad():                                               # move some rows between calls
            for model in (new_model, old_model):
                model.prior_bank.phi_embed[::2].mul_(1.3)
    assert max(flagged_new) > 0


# ------------------------------------------------------------------------------------------------ A246

def test_subnormal_rows_break_the_entrywise_doubling_but_not_the_verdict() -> None:
    g = torch.Generator().manual_seed(11)
    H, d, c = 2, 8, 0.5 * math.log(KAPPA)
    screen = torch.full((H,), 0.5 * (2.0 * c) ** 2 * (1.0 - 1e-4), dtype=torch.float64).to(torch.float32)
    phi = 1e-39 * torch.randn(40, H * d * d, generator=g)
    S0 = _traceless_symmetric_blocks(phi, None, H, d, torch.float32)
    T2 = _doubled_traceless_symmetric(phi, None, H, d, torch.float32)
    assert not torch.equal(2.0 * S0, T2)                                    # gradual underflow: entries differ
    assert torch.equal(_old_screen(2.0 * S0, screen), _old_screen(T2, screen))
    assert torch.equal(_new_screen(2.0 * S0, screen), _new_screen(T2, screen))
    assert not bool(_new_screen(T2, screen).any())


# ------------------------------------------------------------------------------------------------ A169

def _stretch_rows(model: torch.nn.Module, seed: int) -> None:
    r"""Every second row: symmetric traceless head blocks with spectrum on [-2, 2] (spread 4 > log 20), no skew part,
    so the storage margin m = 8 d eps (||X||_F + 2c) stays near the live rows' (a large skew part widens it)."""
    gen = torch.Generator().manual_seed(seed)
    H, d = 2, model.group.irrep_dims[0]
    with torch.no_grad():
        phi = model.prior_bank.phi_embed
        phi.copy_(1e-3 * torch.randn(phi.shape, generator=gen))
        n = phi[1::2].shape[0]
        Q, _ = torch.linalg.qr(torch.randn(n, H, d, d, generator=gen, dtype=torch.float64))
        lam = torch.linspace(-2.0, 2.0, d, dtype=torch.float64).expand(n, H, d)
        phi[1::2].copy_((Q @ torch.diag_embed(lam) @ Q.mT).reshape(n, -1))


@pytest.mark.parametrize("d", [8, 32])
def test_projected_rows_do_not_certify_at_their_center_at_d_8_and_above(d: int,
                                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    r"""Without the stored copies a second call with the table unchanged decomposes the projected rows again."""
    model = _FrameTables(d)
    _stretch_rows(model, 0)
    assert int(bound_phi_frame_condition_(model, KAPPA)) > 0
    snapshot = model.prior_bank.phi_embed.detach().clone()
    blocks = _counting_eigh(monkeypatch)
    assert int(bound_phi_frame_condition_(model, KAPPA)) == 0
    assert sum(blocks) > 0                                                  # the projected rows reach eigh again
    assert torch.equal(model.prior_bank.phi_embed.detach(), snapshot)        # and are left unchanged (fixed point)


@pytest.mark.parametrize("d", [8, 32])
def test_row_lazy_route_skips_unchanged_rows_and_stores_the_same_values(d: int,
                                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    r"""``m_phi_sparse_rows=True``: a second call with the table unchanged makes no eigendecomposition, and over calls
    that move a few rows between them the table equals, bit for bit, the table of the same calls without the stored
    copies."""
    lazy, dense = _FrameTables(d, sparse_rows=True), _FrameTables(d, sparse_rows=False)
    for model in (lazy, dense):
        _stretch_rows(model, 1)
    assert int(bound_phi_frame_condition_(lazy, KAPPA)) == int(bound_phi_frame_condition_(dense, KAPPA)) > 0
    assert torch.equal(lazy.prior_bank.phi_embed.detach(), dense.prior_bank.phi_embed.detach())
    blocks = _counting_eigh(monkeypatch)
    stats: Dict[str, float] = {}
    assert int(bound_phi_frame_condition_(lazy, KAPPA, stats_out=stats)) == 0
    assert blocks == [] and stats["frame_bound_decomposed_rows"] == 0 and stats["frame_bound_flagged_rows"] > 0
    gen = torch.Generator().manual_seed(2)
    for call in range(4):
        touched = torch.randperm(7, generator=gen)[:2]                      # the rows a sparse step moves
        noise = (0.5 if call % 2 else 1e-3) * torch.randn(2, lazy.prior_bank.phi_embed.shape[1], generator=gen)
        for model in (lazy, dense):
            with torch.no_grad():
                model.prior_bank.phi_embed[touched] += noise
        blocks.clear()
        changed_lazy = int(bound_phi_frame_condition_(lazy, KAPPA))
        lazy_blocks = sum(blocks)
        blocks.clear()
        changed_dense = int(bound_phi_frame_condition_(dense, KAPPA))
        assert changed_lazy == changed_dense
        assert torch.equal(lazy.prior_bank.phi_embed.detach(), dense.prior_bank.phi_embed.detach())
        assert lazy_blocks <= sum(blocks)
        assert lazy_blocks <= 2 * 2                                         # at most the touched rows' two heads
        assert float(_exact_spread(lazy.prior_bank.phi_embed, 2, d).max()) <= math.log(KAPPA) * (1 + 1e-6)


def test_row_copies_restart_when_the_budget_changes_or_the_capacity_is_reached() -> None:
    model = _FrameTables(8, sparse_rows=True)
    _stretch_rows(model, 3)
    bound_phi_frame_condition_(model, KAPPA)
    copies = gauge_optim._FRAME_ROW_SNAPSHOTS[model]
    assert copies.used > 0 and copies.budget == math.log(KAPPA)
    bound_phi_frame_condition_(model, 10.0)                                 # a new budget: a new cache
    assert gauge_optim._FRAME_ROW_SNAPSHOTS[model] is not copies
    small = gauge_optim._RowSnapshot(model.prior_bank.phi_embed.detach(), 1.0, capacity=3)
    rows = model.prior_bank.phi_embed.detach()
    small.record(torch.tensor([0, 1]), rows)
    small.record(torch.tensor([2, 3]), rows)                                # 4 > 3: restart, only rows 2, 3 held
    assert small.used == 2
    held = small.unchanged(torch.arange(7), rows)
    assert held.tolist() == [False, False, True, True, False, False, False]


# ------------------------------------------------------------------------------------------------ A189

@pytest.mark.parametrize("overrides", [
    dict(gauge_group="block_sl", embed_dim=8, n_heads=2),
    dict(gauge_group="block_glmc", embed_dim=8, n_heads=2),
    dict(gauge_group="block_glk", embed_dim=8, n_heads=2, gauge_tying="shared"),
    dict(gauge_group="block_glk", embed_dim=9, n_heads=3, gauge_tying="shared"),
])
def test_local_and_shared_head_blocks_match_the_padded_basis(overrides: dict) -> None:
    model = _bound_model(**overrides)
    group = model.group
    H, d = len(group.irrep_dims), group.irrep_dims[0]
    padded = group.head_block_generators().detach().to(torch.float64)          # (n_gen, H, d, d)
    gram = padded.square().sum(dim=(1, 2, 3))
    if group.certified_block_local_basis() is not None:
        basis = _HeadBasis("local", group._local_basis_as(torch.float64, CPU))
    else:
        assert group.shared_elementary_blocks()
        basis = _HeadBasis("shared", None)
    gen = torch.Generator().manual_seed(4)
    phi = torch.randn(20, group.generator_shape[0], generator=gen)
    for dtype, tol in ((torch.float32, 2e-6), (torch.float64, 1e-14)):
        torch.testing.assert_close(_head_blocks(phi, basis, H, d, dtype), _head_blocks(phi, padded, H, d, dtype),
                                   rtol=0.0, atol=tol)
    phi = phi * 2.0                                                          # over the bound
    new, over_new, _ = _frame_bound_rows(phi, basis, None, H, d, 0.5 * math.log(KAPPA), torch.float32)
    old, over_old, _ = _frame_bound_rows(phi, padded, gram, H, d, 0.5 * math.log(KAPPA), torch.float32)
    assert bool(over_new.any()) and torch.equal(over_new, over_old)
    torch.testing.assert_close(new, old, rtol=0.0, atol=2e-5)


@pytest.mark.parametrize("overrides", [
    dict(gauge_group="block_sl", embed_dim=8, n_heads=2),
    dict(gauge_group="block_glmc", embed_dim=8, n_heads=2),
    dict(gauge_group="block_glk", embed_dim=8, n_heads=2, gauge_tying="shared"),
])
def test_bound_on_local_and_shared_bases_matches_the_padded_contraction(overrides: dict) -> None:
    r"""The bound on the whole table equals the projection with the padded head-block basis (the earlier contraction)
    to float32 rounding, on the same rows."""
    model = _bound_model(**overrides)
    group = model.group
    H, d = len(group.irrep_dims), group.irrep_dims[0]
    gen = torch.Generator().manual_seed(5)
    with torch.no_grad():
        phi = model.prior_bank.phi_embed
        phi.copy_(torch.randn(phi.shape, generator=gen) * torch.logspace(-1, 0.6, phi.shape[0])[:, None])
    before = phi.detach().clone()
    padded = group.head_block_generators().detach().to(torch.float64)
    new, over, failed = _frame_bound_rows(before, padded, padded.square().sum(dim=(1, 2, 3)), H, d,
                                          0.5 * math.log(KAPPA), torch.float32, store_dtype=before.dtype)
    assert not bool(failed.any()) and bool(over.any())
    reference = torch.where(over[:, None], new.to(before.dtype), before)
    changed = int(bound_phi_frame_condition_(model, KAPPA))
    assert changed == int(over.sum())
    torch.testing.assert_close(phi.detach(), reference, rtol=0.0, atol=2e-5)
    assert torch.equal(phi.detach()[~over], before[~over])


def test_shared_elementary_certificate() -> None:
    shared = get_group("block_glk")(8, 2, tying="shared")
    assert shared.shared_elementary_blocks()
    assert not get_group("block_glk")(8, 2).shared_elementary_blocks()
    assert not get_group("block_sl")(8, 2).shared_elementary_blocks()
    with torch.no_grad():
        shared.generators[0].mul_(2.0)                                      # a mutated basis is checked again
    assert not shared.shared_elementary_blocks()


@pytest.mark.parametrize("n_heads, exact", [(2, True), (3, False)])
def test_shared_block_glk_transport_embeds_by_reshape(n_heads: int, exact: bool,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = test_config(vocab_size=7, embed_dim=3 * n_heads, n_heads=n_heads, max_seq_len=4, gauge_group="block_glk",
                      gauge_tying="shared", phi_scale=0.3)
    tokens, targets = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 1]]), torch.tensor([[1, 2, 3, 4], [5, 6, 1, 2]])

    def run() -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        torch.manual_seed(0)
        model = VFEModel(cfg)
        with torch.no_grad():
            logits = model(tokens)[0]
        loss = model(tokens, targets)[1]
        loss.backward()
        return logits, {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    new_logits, new_grads = run()
    with monkeypatch.context() as m:
        m.setattr(GaugeGroup, "shared_elementary_blocks", lambda self: False)   # the padded contraction
        old_logits, old_grads = run()
    assert new_grads.keys() == old_grads.keys() and "prior_bank.phi_embed" in new_grads
    if exact:
        assert torch.equal(new_logits, old_logits)
        for name in new_grads:
            assert torch.equal(new_grads[name], old_grads[name]), name
    else:
        torch.testing.assert_close(new_logits, old_logits, rtol=1e-5, atol=1e-6)
        for name in new_grads:
            torch.testing.assert_close(new_grads[name], old_grads[name], rtol=1e-4, atol=1e-6)


# ------------------------------------------------------------------------------------------------ A204

class _EmaModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.table = torch.nn.Parameter(torch.zeros(6, 5))
        self.bias = torch.nn.Parameter(torch.zeros(5, dtype=torch.float64))
        self.frozen = torch.nn.Parameter(torch.ones(2), requires_grad=False)


def _ema_trajectory(staged: bool, monkeypatch: pytest.MonkeyPatch) -> List[Any]:
    from vfe3.weight_ema import WeightEMA
    model = _EmaModel()
    ema = WeightEMA(model, decay=0.9, start_step=1, device="cpu")
    if staged:                                                              # the host route, forced for CPU parameters
        monkeypatch.setattr(ema, "_host_staged", lambda: True)
    gen = torch.Generator().manual_seed(9)
    record: List[Any] = []
    for step in range(1, 7):
        with torch.no_grad():
            model.table.add_(torch.randn(6, 5, generator=gen))
            model.bias.add_(torch.randn(5, generator=gen, dtype=torch.float64))
        ema.observe(step, accepted=step != 4)
        if step == 3:
            raw = [p.detach().clone() for p in (model.table, model.bias)]
            with ema.applied():
                record.append([p.detach().clone() for p in (model.table, model.bias)])
            assert all(torch.equal(a, b) for a, b in zip(raw, (model.table, model.bias)))
        if step == 5:
            record.append([s.clone() for s in ema.state_dict()["shadow"]])
    record.append([s.clone() for s in ema.shadow_state().values()])
    assert ema.updates == 5 and ema._worker is None
    return record


def test_host_route_of_the_weight_ema_equals_the_synchronous_update(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""The staged copy plus worker-thread lerp gives the synchronous update's shadow bit for bit, and every read (the
    swap, the checkpoint record, the shadow) joins the update in flight first."""
    synchronous = _ema_trajectory(False, monkeypatch)
    staged = _ema_trajectory(True, monkeypatch)
    assert len(synchronous) == len(staged) == 3
    for a, b in zip(synchronous, staged):
        assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_host_route_error_is_raised_by_the_next_join(monkeypatch: pytest.MonkeyPatch) -> None:
    from vfe3.weight_ema import WeightEMA
    model = _EmaModel()
    ema = WeightEMA(model, decay=0.9, start_step=1, device="cpu")
    monkeypatch.setattr(ema, "_host_staged", lambda: True)
    ema.observe(1, True)

    def failing(*args: Any, **kwargs: Any) -> None:
        raise ValueError("lerp failed")

    monkeypatch.setattr(torch, "_foreach_lerp_", failing)
    ema.observe(2, True)                                                    # the worker hits the error
    with pytest.raises(RuntimeError, match="host update failed"):
        ema.state_dict()


def test_cuda_weight_ema_host_route_matches_a_hand_ema() -> None:
    import os
    if os.environ.get("VFE3_TEST_DEVICE", "cpu") != "cuda" or not torch.cuda.is_available():
        pytest.skip("CUDA lane only")
    from vfe3.weight_ema import WeightEMA
    model = _EmaModel().cuda()
    ema = WeightEMA(model, decay=0.8, start_step=1, device="cpu")
    assert ema._host_staged()
    gen = torch.Generator(device="cuda").manual_seed(3)
    hand = None
    for step in range(1, 5):
        with torch.no_grad():
            model.table.add_(torch.randn(6, 5, generator=gen, device="cuda"))
        ema.observe(step, True)
        value = model.table.detach().cpu()
        hand = value.clone() if hand is None else torch.lerp(hand, value, 1.0 - 0.8)
    assert all(buffer.is_pinned() for buffer in ema._staging)
    assert torch.equal(ema.shadow_state()["table"], hand)
