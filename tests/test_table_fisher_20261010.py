r"""Natural-gradient M-step for the type-prior tables (m_table_update_mode='fisher'; audit 2026-10-10, item 22).

Pins, on tiny CPU shapes: the default builds plain AdamW; the step equals the closed form
(mu, l) -= eta (N/n_v) (s grad_mu, 2 grad_l) with AdamW's decoupled decay; unvisited rows take no gradient
step; the per-row Fisher norm is bounded by the trust radius; the LR schedule scales the step; one step at
eta = 1 on a self-coupling objective lands on the closed-form M-step mean; the validator refuses a tied
prior-bank decode and the block-Cholesky table; and a short training run is finite and logs the step health.
"""

import csv
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from tests.profiles import test_config
from vfe3.data.datasets import TokenWindows
from vfe3.model.model import VFEModel
from vfe3.run_artifacts import RunArtifacts
from vfe3.table_optim import TABLE_FISHER_DIAGNOSTIC_KEYS, TableFisherAdamW, TableFisherStatistics
from vfe3.train import build_optimizer, train


def _cfg(**overrides: object):
    return test_config(**{**dict(vocab_size=9, embed_dim=4, n_heads=2, max_seq_len=6, n_layers=1,
                                 decode_mode="linear", m_table_update_mode="fisher"), **overrides})


def _tables(model: VFEModel):
    return model.prior_bank.mu_embed, model.prior_bank.sigma_log_embed


def _fill_grads(model: VFEModel, seed: int = 0) -> None:
    g = torch.Generator().manual_seed(seed)
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.randn(p.shape, generator=g, dtype=p.dtype)


def test_default_mode_builds_plain_adamw() -> None:
    cfg = test_config(vocab_size=9, embed_dim=4, n_heads=2, max_seq_len=6, n_layers=1)
    assert type(build_optimizer(VFEModel(cfg), cfg)) is torch.optim.AdamW
    assert "m_table_fisher_lr" in cfg.inactive_fields()


def test_step_equals_the_closed_form_natural_gradient() -> None:
    cfg = _cfg(m_table_fisher_lr=0.3, m_table_fisher_trust_radius=1e9)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    optimizer = build_optimizer(model, cfg)
    assert isinstance(optimizer, TableFisherAdamW)
    mu, log_s = _tables(model)
    _fill_grads(model)
    tokens = torch.tensor([[1, 2, 2, 5, 1, 1]])
    scored = torch.tensor(6)
    count = torch.bincount(tokens.reshape(-1), minlength=cfg.vocab_size).double()
    groups = {g.get("table_fisher"): g for g in optimizer.param_groups}
    s = log_s.detach().double().exp().clamp(min=model.prior_bank.eps)
    weight = (6.0 / count.clamp_min(1.0)).unsqueeze(-1)
    active = (count > 0).double().unsqueeze(-1)
    expected = {}
    for name, p, direction in (("mu", mu, s * mu.grad.double()), ("sigma", log_s, 2.0 * log_s.grad.double())):
        decay = 1.0 - groups[name]["lr"] * groups[name]["weight_decay"]
        expected[name] = p.detach().double() * decay - 0.3 * active * weight * direction
    other = [p.detach().clone() for p in model.parameters() if p is not mu and p is not log_s]
    optimizer.table_statistics_provider.set_batch(tokens, scored)
    optimizer.step()
    torch.testing.assert_close(mu.detach().double(), expected["mu"], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(log_s.detach().double(), expected["sigma"], rtol=1e-6, atol=1e-7)
    after = [p.detach() for p in model.parameters() if p is not mu and p is not log_s]
    assert any(not torch.equal(a, b) for a, b in zip(other, after))      # AdamW stepped the rest


def test_trust_radius_bounds_each_row_and_the_schedule_scales_it() -> None:
    cfg = _cfg(m_table_fisher_lr=10.0, m_table_fisher_trust_radius=0.05, weight_decay=0.0,
               mu_weight_decay=0.0, sigma_weight_decay=0.0)
    for factor in (1.0, 0.5):
        torch.manual_seed(0)
        model = VFEModel(cfg)
        optimizer = build_optimizer(model, cfg)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: factor)
        del scheduler
        mu, log_s = _tables(model)
        before_mu, before_l = mu.detach().clone(), log_s.detach().clone()
        s = before_l.exp()
        _fill_grads(model)
        optimizer.table_statistics_provider.set_batch(torch.tensor([[1, 2, 3, 3, 4, 1]]), torch.tensor(6))
        optimizer.step()
        d_mu, d_l = mu.detach() - before_mu, log_s.detach() - before_l
        norm = ((d_mu.square() / s).sum(-1) + 0.5 * d_l.square().sum(-1)).sqrt()
        visited = torch.tensor([1, 2, 3, 4])
        torch.testing.assert_close(norm[visited], torch.full((4,), 0.05 * factor), rtol=1e-5, atol=0.0)
        unvisited = torch.tensor([0, 5, 6, 7, 8])
        assert torch.equal(norm[unvisited], torch.zeros(5))


class _Bank(torch.nn.Module):
    def __init__(self, log_s: torch.Tensor) -> None:
        super().__init__()
        self.log_s, self.eps = log_s, 1e-12

    def _prior_sigma_log_table(self) -> torch.Tensor:
        return self.log_s


def test_unit_step_on_the_self_coupling_objective_is_the_closed_form_m_step() -> None:
    r"""With L = (1/N) sum_i KL(N(m_i, c_i) || p_{x_i}) the metric (n_v/N) G_v is the Hessian in mu, so eta = 1
    moves mu_v to the mean of the m_i of type v in one step."""
    g = torch.Generator().manual_seed(3)
    V, K = 4, 3
    mu = torch.nn.Parameter(torch.randn(V, K, generator=g, dtype=torch.float64))
    log_s = torch.randn(V, K, generator=g, dtype=torch.float64)
    x = torch.tensor([0, 2, 2, 3, 0, 0, 2])
    m = torch.randn(x.shape[0], K, generator=g, dtype=torch.float64)
    c = torch.rand(x.shape[0], K, generator=g, dtype=torch.float64) + 0.5
    s = log_s.exp()
    kl = 0.5 * ((c + (m - mu[x]).square()) / s[x] - 1.0 + log_s[x] - c.log()).sum(-1)
    kl.mean().backward()
    optimizer = TableFisherAdamW([{"params": [mu], "lr": 1.0, "weight_decay": 0.0, "table_fisher": "mu"}],
                                 TableFisherStatistics(_Bank(log_s)), table_fisher_lr=1.0,
                                 table_fisher_trust_radius=1e9)
    optimizer.table_statistics_provider.set_batch(x.unsqueeze(0), torch.tensor(x.shape[0]))
    before = mu.detach().clone()
    optimizer.step()
    for v in range(V):
        want = m[x == v].mean(0) if bool((x == v).any()) else before[v]
        torch.testing.assert_close(mu.detach()[v], want, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("overrides", [
    dict(decode_mode="diagonal_chunked", untie_decode_bank=False),
    dict(prior_covariance_table="block_cholesky"),
])
def test_validator_refuses_tables_the_metric_does_not_describe(overrides: dict) -> None:
    with pytest.raises(ValueError, match="m_table_update_mode='fisher'|prior_covariance_table"):
        _cfg(**overrides)


def test_short_training_run_is_finite_and_logs_the_step_health(tmp_path: Path) -> None:
    cfg = _cfg(warmup_steps=1, max_steps=3, log_interval=1, eval_interval=0, checkpoint_interval=0)
    stream = torch.randint(0, 9, (6 * 8 + 1,), generator=torch.Generator().manual_seed(0))
    loader = DataLoader(TokenWindows(stream, 6), batch_size=4, shuffle=False, drop_last=True)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    mu0 = model.prior_bank.mu_embed.detach().clone()
    artifacts = RunArtifacts(tmp_path / "run", cfg, model)
    losses = train(model, loader, cfg, n_steps=3, device=torch.device("cpu"), artifacts=artifacts,
                   log_interval=1, eval_interval=0)
    assert len(losses) == 3 and all(torch.isfinite(torch.tensor(losses)))
    assert not torch.equal(model.prior_bank.mu_embed.detach(), mu0)
    with open(artifacts.csv_path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows and all(key in rows[0] for key in TABLE_FISHER_DIAGNOSTIC_KEYS)
    assert any(float(row["table_fisher_active_rows"]) > 0 for row in rows if row["table_fisher_active_rows"])


@pytest.mark.parametrize("phi_mode", ["pullback_group", "fisher_group"])
def test_composes_with_the_manifold_phi_optimizer(phi_mode: str) -> None:
    from vfe3.table_optim import TableFisherGaugeManifoldAdamW
    cfg = _cfg(m_phi_update_mode=phi_mode, m_table_fisher_trust_radius=1e9, weight_decay=0.0,
               mu_weight_decay=0.0, phi_weight_decay=0.0)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    optimizer = build_optimizer(model, cfg)
    assert isinstance(optimizer, TableFisherGaugeManifoldAdamW)
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
    _, loss, _ = model(tokens[:, :-1], tokens[:, 1:])
    loss.backward()
    mu = model.prior_bank.mu_embed
    s = model.prior_bank.sigma_log_embed.detach().exp()
    expected = mu.detach() - 0.1 * (5.0 / torch.bincount(tokens[0, :-1], minlength=9).clamp_min(1)
                                    .unsqueeze(-1)) * s * mu.grad * (torch.bincount(tokens[0, :-1], minlength=9) > 0
                                                                     ).unsqueeze(-1)
    phi_before = model.prior_bank.phi_embed.detach().clone()
    if optimizer.phi_statistics_provider is not None:
        optimizer.phi_statistics_provider.set_batch(tokens[:, :-1], torch.tensor(5))
    optimizer.table_statistics_provider.set_batch(tokens[:, :-1], torch.tensor(5))
    optimizer.step()
    torch.testing.assert_close(mu.detach(), expected, rtol=1e-5, atol=1e-7)
    assert not torch.equal(model.prior_bank.phi_embed.detach(), phi_before)
