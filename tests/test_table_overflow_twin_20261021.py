r"""The float64 overflow twin of the table Fisher step runs on the candidate rows only (audit 2026-10-20 item 44).

Before the change ``_robust_table_steps`` ran over all :math:`V\times K` rows and its result was read only where the
row's Fisher norm overflowed. It now runs on the step's input ids (a superset of the active rows, of host-known size),
and the result is scattered back. Pinned here: bitwise equality with the full-table twin (``_overflow_candidate_rows``
patched to every row) on overflowed and ordinary rows, with and without the step diagnostics, for the diagonal step,
the frame-diagonal step (``gaussian_frame_diagonal``) and alongside the decode-bank Fisher step
(``m_decode_update_mode='fisher'``), whose float64 step is its only path and therefore still runs on every row; and
that the twin sees fewer than :math:`V` rows when the step has fewer input ids than the table has rows.
"""
from __future__ import annotations

import math
import warnings
from typing import Any, Callable, Dict, List, Tuple

import pytest
import torch

import vfe3.table_optim as table_optim
from tests.profiles import test_config
from tests.test_frame_diagonal_table_fisher_20261018 import _cfg as _frame_cfg, _model as _frame_model
from vfe3.model.model import VFEModel
from vfe3.table_optim import TableFisherAdamW, TableFisherStatistics
from vfe3.train import build_optimizer


@pytest.fixture(autouse=True)
def _quiet_two_threads():
    torch.set_num_threads(2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _every_row(tokens: torch.Tensor, V: int, device: torch.device) -> torch.Tensor:
    return torch.arange(V, device=device)                                   # the pre-item-44 full-table twin


def _spy(monkeypatch: pytest.MonkeyPatch) -> List[int]:
    rows: List[int] = []
    original = table_optim._robust_table_steps

    def spy(g_mu, g_l, weight, *args, **kwargs):
        rows.append(int(weight.shape[0]))
        return original(g_mu, g_l, weight, *args, **kwargs)

    monkeypatch.setattr(table_optim, "_robust_table_steps", spy)
    return rows


def _compare(run: Callable[[], Tuple[Dict[str, torch.Tensor], Dict[str, float]]],
             monkeypatch: pytest.MonkeyPatch) -> Tuple[Dict[str, torch.Tensor], List[int]]:
    rows = _spy(monkeypatch)
    lean, lean_diag = run()
    lean_rows = list(rows)
    with monkeypatch.context() as m:
        m.setattr(table_optim, "_overflow_candidate_rows", _every_row)
        full, full_diag = run()
    assert lean.keys() == full.keys()
    for name in lean:
        assert torch.equal(lean[name], full[name]), name                     # bitwise, overflow and ordinary rows
    assert lean_diag.keys() == full_diag.keys()
    for key in lean_diag:
        a, b = lean_diag[key], full_diag[key]
        assert (math.isnan(a) and math.isnan(b)) or a == b, key
    return lean, lean_rows


# ------------------------------------------------------------------------------------------------ diagonal step

class _Bank(torch.nn.Module):
    def __init__(self, log_s: torch.Tensor) -> None:
        super().__init__()
        self.log_s, self.eps = log_s, 1e-12

    def _prior_sigma_log_table(self) -> torch.Tensor:
        return self.log_s


@pytest.mark.parametrize("collect", [False, True])
@pytest.mark.parametrize("schedule", [1.0, 0.5])
def test_diagonal_step_is_bitwise_the_full_table_twin(collect: bool, schedule: float,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    V, K, radius = 11, 4, 0.5
    gen = torch.Generator().manual_seed(0)
    mu0 = torch.randn(V, K, generator=gen)
    l0 = 0.3 * torch.randn(V, K, generator=gen)
    log_s = torch.log(torch.rand(V, K, generator=gen) * 4.0 + 1.0)
    g_mu = torch.randn(V, K, generator=gen) * 1e-2
    g_l = torch.randn(V, K, generator=gen) * 1e-2
    g_mu[0], g_mu[3] = 1e17, -1e35                                          # norm_sq overflow, step overflow
    g_l[5] = 2e30                                                           # overflow through the log-variance part
    tokens = torch.tensor([[0, 1, 3, 3], [5, 7, 1, 0]])                     # 8 ids < V = 11; rows 2, 4, 6, 8-10 absent

    def run() -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        mu, log_v = torch.nn.Parameter(mu0.clone()), torch.nn.Parameter(l0.clone())
        optimizer = TableFisherAdamW(
            [{"params": [mu], "lr": 1.0 * schedule, "initial_lr": 1.0, "weight_decay": 0.01, "table_fisher": "mu"},
             {"params": [log_v], "lr": 1.0 * schedule, "initial_lr": 1.0, "weight_decay": 0.0,
              "table_fisher": "sigma"}],
            TableFisherStatistics(_Bank(log_s)), table_fisher_lr=1.0, table_fisher_trust_radius=radius)
        optimizer._collect_table_diag = collect
        mu.grad, log_v.grad = g_mu.clone(), g_l.clone()
        optimizer.table_statistics_provider.set_batch(tokens, torch.tensor(8192))
        optimizer.step()
        return {"mu": mu.detach().clone(), "log_s": log_v.detach().clone()}, dict(optimizer._table_diag)

    lean, rows = _compare(run, monkeypatch)
    assert rows == [tokens.numel()]                                         # the twin saw 8 rows, not 11
    assert bool(torch.isfinite(lean["mu"]).all()) and bool(torch.isfinite(lean["log_s"]).all())
    for row in (0, 3, 5):                                                   # overflowed rows take the full trust step
        d_mu = lean["mu"][row].double() - mu0[row].double() * (1.0 - 0.01 * schedule)
        d_l = lean["log_s"][row].double() - l0[row].double()
        norm = ((d_mu.square() / log_s[row].double().exp()).sum() + 0.5 * d_l.square().sum()).sqrt()
        assert float(norm) == pytest.approx(schedule * radius, rel=1e-4)


def test_candidates_are_every_row_when_the_step_has_at_least_v_ids() -> None:
    tokens = torch.tensor([[2, 2, 0], [1, 1, 1]])
    assert torch.equal(table_optim._overflow_candidate_rows(tokens, 4, torch.device("cpu")), torch.arange(4))
    assert torch.equal(table_optim._overflow_candidate_rows(tokens, 9, torch.device("cpu")), tokens.reshape(-1))


# ------------------------------------------------------------------------------------------------ frame-diagonal step

def test_frame_diagonal_step_is_bitwise_the_full_table_twin(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _frame_cfg(m_table_fisher_lr=0.3, m_table_fisher_trust_radius=0.5)
    reference = _frame_model(cfg).float()                                  # float32, frames far from the identity
    tokens = torch.tensor([[1, 2, 2], [1, 3, 7]])                           # 6 ids < V = 9
    gen = torch.Generator().manual_seed(3)
    grads = {name: torch.randn(p.shape, generator=gen) * 1e-2 for name, p in reference.named_parameters()
             if p.requires_grad}
    grads["prior_bank.mu_embed"][2] = 1e20                                  # (eta N/n s g)^2 overflows float32
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
        pb = model.prior_bank
        return ({"mu": pb.mu_embed.detach().clone(), "log_s": pb.sigma_log_embed.detach().clone()},
                dict(optimizer._table_diag))

    lean, rows = _compare(run, monkeypatch)
    assert rows == [tokens.numel()]
    assert bool(torch.isfinite(lean["mu"]).all()) and bool(torch.isfinite(lean["log_s"]).all())


# ------------------------------------------------------------------------------------------------ decode-bank step

def test_decode_bank_step_with_overflow_rows_is_unchanged_and_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = test_config(vocab_size=9, embed_dim=4, n_heads=2, max_seq_len=6, n_layers=1, decode_mode="diagonal_chunked",
                      untie_decode_bank=True, m_decode_update_mode="fisher", m_table_update_mode="fisher",
                      m_table_fisher_lr=1.0, m_table_fisher_trust_radius=0.05, decode_mu_weight_decay=0.0,
                      sigma_weight_decay=0.0, mu_weight_decay=0.0)
    tokens = torch.tensor([[1, 2, 3], [4, 4, 1]])                           # 6 ids < V = 9
    targets = torch.tensor([[2, 3, 4], [4, 1, -100]])
    torch.manual_seed(0)
    initial = VFEModel(cfg).state_dict()
    gen = torch.Generator().manual_seed(5)
    grads = {name: torch.randn(p.shape, generator=gen) * 1e-2 for name, p in VFEModel(cfg).named_parameters()
             if p.requires_grad}
    for name in ("prior_bank.mu_embed", "prior_bank.decode_mu_embed"):
        grads[name][1] = 1e35
    grads["prior_bank.decode_sigma_log_embed"][6] = 1e30                    # a non-target decode row

    def run() -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        model = VFEModel(cfg)
        model.load_state_dict(initial)
        optimizer = build_optimizer(model, cfg)
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.grad = grads[name].clone()
        optimizer._collect_table_diag = True
        optimizer.table_statistics_provider.set_batch(tokens, torch.tensor(5), targets=targets)
        optimizer.step()
        pb = model.prior_bank
        return ({n: getattr(pb, n).detach().clone() for n in
                 ("mu_embed", "sigma_log_embed", "decode_mu_embed", "decode_sigma_log_embed")},
                dict(optimizer._table_diag))

    lean, rows = _compare(run, monkeypatch)
    assert rows == [tokens.numel(), cfg.vocab_size]                         # encoder twin on 6 rows; decode on all V
    pb0 = {k.split(".")[-1]: v for k, v in initial.items() if k.startswith("prior_bank.")}
    t = pb0["decode_sigma_log_embed"].double().exp()
    dm = lean["decode_mu_embed"].double() - pb0["decode_mu_embed"].double()
    dl = lean["decode_sigma_log_embed"].double() - pb0["decode_sigma_log_embed"].double()
    norm = ((dm.square() / t).sum(-1) + 0.5 * dl.square().sum(-1)).sqrt()                      # (V,)
    assert bool(torch.isfinite(norm).all())
    assert float(norm[1]) == pytest.approx(0.05, rel=1e-5) and float(norm[6]) == pytest.approx(0.05, rel=1e-5)
