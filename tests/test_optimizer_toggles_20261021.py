r"""Opt-in optimizer toggles of the 2026-10-20 audit remediation (lane L3; audit items 2, 10, 49; M2, M5, M6).

``m_phi_sparse_rows=True`` steps ``phi_embed`` with row-lazy AdamW: a row that is neither among the step's input tokens
nor carries a nonzero gradient takes no moment update, no step and no decoupled decay, and a stepped row's bias
correction uses its own step count :math:`k_v`. Pinned here: equivalence to dense AdamW when every row is touched every
step; untouched rows exactly unchanged; the per-row bias correction against a hand computation; the token-occurrence
mask; a checkpoint round trip of the per-row counts; a tiny end-to-end ``train_step``.

``m_decode_update_mode='fisher'`` steps the untied decode bank with the natural gradient of the Gaussian Fisher metric
:math:`(c_v/N)\operatorname{diag}(1/t_v, 1/2)`. Pinned here: the closed-form step, the trust radius and schedule, the
validator, the default topology, and a tiny end-to-end ``train_step``.
"""

import math
from pathlib import Path

import pytest
import torch

from tests.profiles import test_config
from vfe3.model.model import VFEModel
from vfe3.run_artifacts import RunArtifacts, load_checkpoint
from vfe3.sparse_row_optim import RowLazyAdamW, RowLazyTableFisherAdamW
from vfe3.table_optim import TableFisherAdamW, TableFisherStatistics
from vfe3.train import build_optimizer, lr_lambda, train_step


# ---------------------------------------------------------------------------------------------------------------------
# m_phi_sparse_rows: row-lazy AdamW
# ---------------------------------------------------------------------------------------------------------------------

class _Tokens:
    r"""Minimal step-statistics provider: the input tokens of the step, as ``TableFisherStatistics`` supplies them."""

    def __init__(self, tokens: "torch.Tensor | None") -> None:
        self.tokens = tokens

    def step_tokens(self) -> "torch.Tensor | None":
        return self.tokens


def _pair(V: int = 5, P: int = 3, lr: float = 0.01, wd: float = 0.1, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    init = torch.randn(V, P, generator=g, dtype=torch.float64)
    lazy_w = torch.nn.Parameter(init.clone())
    dense_w = torch.nn.Parameter(init.clone())
    lazy = RowLazyAdamW([{"params": [lazy_w], "lr": lr, "weight_decay": wd, "sparse_rows": True}])
    dense = torch.optim.AdamW([{"params": [dense_w], "lr": lr, "weight_decay": wd}], foreach=False)
    return lazy_w, lazy, dense_w, dense


def test_row_lazy_equals_dense_adamw_when_every_row_is_touched() -> None:
    lazy_w, lazy, dense_w, dense = _pair()
    g = torch.Generator().manual_seed(1)
    for step in range(6):
        grad = torch.randn(lazy_w.shape, generator=g, dtype=torch.float64)
        for opt, w in ((lazy, lazy_w), (dense, dense_w)):
            opt.param_groups[0]["lr"] = 0.01 * (1.0 - 0.1 * step)          # a moving schedule
            w.grad = grad.clone()
            opt.step()
        torch.testing.assert_close(lazy_w.detach(), dense_w.detach(), rtol=1e-12, atol=1e-14)
    state = lazy.state[lazy_w]
    assert torch.equal(state["row_step"], torch.full((5,), 6.0, dtype=torch.float64))
    assert float(state["step"]) == 6.0
    torch.testing.assert_close(state["exp_avg"], dense.state[dense_w]["exp_avg"], rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(state["exp_avg_sq"], dense.state[dense_w]["exp_avg_sq"], rtol=1e-12, atol=1e-14)


def test_untouched_rows_are_exactly_unchanged() -> None:
    lazy_w, lazy, dense_w, dense = _pair(wd=0.5)
    before = lazy_w.detach().clone()
    g = torch.Generator().manual_seed(2)
    for step in range(4):
        grad = torch.randn(lazy_w.shape, generator=g, dtype=torch.float64)
        grad[2:4] = 0.0                                                    # rows 2, 3 never touched
        if step >= 1:
            grad[0] = 0.0                                                  # row 0 touched only at step 1
        for opt, w in ((lazy, lazy_w), (dense, dense_w)):
            w.grad = grad.clone()
            opt.step()
    assert torch.equal(lazy_w.detach()[2:4], before[2:4])                  # no decay, no step
    assert not torch.equal(dense_w.detach()[2:4], before[2:4])             # dense AdamW decays them
    state = lazy.state[lazy_w]
    assert torch.equal(state["row_step"], torch.tensor([1.0, 4.0, 0.0, 0.0, 4.0], dtype=torch.float64))
    assert torch.equal(state["exp_avg"][2:4], torch.zeros(2, 3, dtype=torch.float64))
    assert torch.equal(state["exp_avg_sq"][2:4], torch.zeros(2, 3, dtype=torch.float64))
    # Row 0 stopped after its one touch: under dense AdamW its momentum tail keeps moving it.
    lazy_row0 = lazy_w.detach()[0].clone()
    for opt, w in ((lazy, lazy_w), (dense, dense_w)):
        grad = torch.zeros_like(w)
        grad[1] = 1.0
        w.grad = grad
        opt.step()
    assert torch.equal(lazy_w.detach()[0], lazy_row0)


def test_per_row_bias_correction_matches_a_hand_computation() -> None:
    lr, wd, b1, b2, eps = 0.02, 0.1, 0.9, 0.999, 1e-8
    w = torch.nn.Parameter(torch.tensor([[0.3, -0.2], [1.0, 2.0]], dtype=torch.float64))
    opt = RowLazyAdamW([{"params": [w], "lr": lr, "weight_decay": wd, "sparse_rows": True}])
    g1 = torch.tensor([0.5, -1.5], dtype=torch.float64)
    g3 = torch.tensor([-0.25, 2.0], dtype=torch.float64)
    schedule = {1: g1, 2: None, 3: g3}                                     # row 0 touched at global steps 1 and 3
    for step in (1, 2, 3):
        grad = torch.zeros(2, 2, dtype=torch.float64)
        grad[1] = 1.0                                                      # row 1 every step
        if schedule[step] is not None:
            grad[0] = schedule[step]
        w.grad = grad
        opt.step()
    # Hand computation for row 0 with its own count k = 1, 2.
    p = torch.tensor([0.3, -0.2], dtype=torch.float64)
    m = torch.zeros(2, dtype=torch.float64)
    v = torch.zeros(2, dtype=torch.float64)
    for k, g in ((1, g1), (2, g3)):
        p = p * (1.0 - lr * wd)
        m = b1 * m + (1.0 - b1) * g
        v = b2 * v + (1.0 - b2) * g * g
        m_hat = m / (1.0 - b1 ** k)
        v_hat = v / (1.0 - b2 ** k)
        p = p - lr * m_hat / (v_hat.sqrt() + eps)
    torch.testing.assert_close(w.detach()[0], p, rtol=1e-12, atol=1e-14)
    assert opt.state[w]["row_step"].tolist() == [2.0, 3.0]


def test_token_occurrence_marks_a_zero_gradient_row_as_touched() -> None:
    w = torch.nn.Parameter(torch.ones(4, 2, dtype=torch.float64))
    provider = _Tokens(None)
    opt = RowLazyAdamW([{"params": [w], "lr": 0.1, "weight_decay": 0.0, "sparse_rows": True}],
                       statistics=provider)
    grad = torch.zeros(4, 2, dtype=torch.float64)
    grad[0] = 1.0
    w.grad = grad.clone()
    provider.tokens = torch.tensor([[0, 1]])
    opt.step()                                                             # rows 0 (grad) and 1 (token)
    w.grad = grad.clone() * 0.0
    provider.tokens = torch.tensor([[0, 0]])
    before = w.detach().clone()
    opt.step()                                                             # row 0 occurs with zero gradient
    assert opt.state[w]["row_step"].tolist() == [2.0, 1.0, 0.0, 0.0]
    assert not torch.equal(w.detach()[0], before[0])                      # its momentum tail moves it
    assert torch.equal(w.detach()[1:], before[1:])


def test_default_and_validator() -> None:
    cfg = test_config(vocab_size=9, embed_dim=4, n_heads=2, max_seq_len=6, n_layers=1)
    optimizer = build_optimizer(VFEModel(cfg), cfg)
    assert type(optimizer) is torch.optim.AdamW
    assert not any(group.get("sparse_rows") for group in optimizer.param_groups)
    with pytest.raises(ValueError, match="m_phi_sparse_rows"):
        test_config(m_phi_sparse_rows=True, m_phi_update_mode="pullback_group")
    with pytest.raises(ValueError, match="m_phi_sparse_rows"):
        test_config(m_phi_sparse_rows=1)
    assert "m_phi_sparse_rows" in test_config(gauge_transport="off").inactive_fields()
    assert "m_phi_sparse_rows" not in cfg.inactive_fields()


def _lazy_cfg(**overrides: object):
    return test_config(**{**dict(vocab_size=9, embed_dim=4, n_heads=2, max_seq_len=6, n_layers=1,
                                 m_phi_sparse_rows=True, phi_weight_decay=0.1, m_phi_lr=0.01,
                                 warmup_steps=1, max_steps=4), **overrides})


def _batch():
    tokens = torch.tensor([[1, 2, 3, 2, 1, 3], [3, 1, 2, 2, 3, 1]])      # types 0, 4..8 never occur
    targets = torch.tensor([[2, 3, 2, 1, 3, 1], [1, 2, 2, 3, 1, -100]])
    return tokens, targets


def test_train_step_leaves_absent_frame_rows_unchanged() -> None:
    tokens, targets = _batch()
    absent = [0, 4, 5, 6, 7, 8]
    present = [1, 2, 3]
    results = {}
    for sparse in (True, False):
        torch.manual_seed(0)
        cfg = _lazy_cfg(m_phi_sparse_rows=sparse)
        model = VFEModel(cfg)
        optimizer = build_optimizer(model, cfg)
        assert isinstance(optimizer, RowLazyAdamW) == sparse
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_lambda(s, cfg))
        phi = model.prior_bank.phi_embed
        before = phi.detach().clone()
        for _ in range(2):
            loss = train_step(model, optimizer, scheduler, tokens, targets, grad_clip=1.0)
            assert math.isfinite(loss)
        results[sparse] = (before, phi.detach().clone(), optimizer)
    before, after, optimizer = results[True]
    assert torch.equal(after[absent], before[absent])
    assert not torch.equal(after[present], before[present])
    state = optimizer.state[optimizer.param_groups[[g.get("role") for g in optimizer.param_groups].index("phi")]
                            ["params"][0]]
    assert state["row_step"][present].tolist() == [2.0, 2.0, 2.0]
    assert state["row_step"][absent].tolist() == [0.0] * len(absent)
    dense_before, dense_after, _ = results[False]
    assert not torch.equal(dense_after[absent], dense_before[absent])      # dense decay reaches them


def test_frame_condition_bound_composes_with_the_row_lazy_step() -> None:
    tokens, targets = _batch()
    torch.manual_seed(0)
    cfg = _lazy_cfg(phi_frame_condition_max=1.5, m_phi_lr=0.5)           # a tight bound the large rate reaches
    model = VFEModel(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_lambda(s, cfg))
    phi = model.prior_bank.phi_embed
    before = phi.detach().clone()
    counter = torch.zeros(2, dtype=torch.int64)
    for _ in range(3):
        assert math.isfinite(train_step(model, optimizer, scheduler, tokens, targets, grad_clip=1.0,
                                        frame_projection_out=counter))
    assert torch.equal(phi.detach()[[0, 4, 5, 6, 7, 8]], before[[0, 4, 5, 6, 7, 8]])
    H, d = cfg.n_heads, cfg.embed_dim // cfg.n_heads
    X = phi.detach()[[1, 2, 3]].double().view(3, H, d, d)                                     # (3, H, d, d)
    S = 0.5 * (X + X.transpose(-1, -2))
    S0 = S - torch.diagonal(S, dim1=-2, dim2=-1).mean(-1)[..., None, None] * torch.eye(d, dtype=torch.float64)
    eigenvalues = torch.linalg.eigvalsh(S0)                                                  # (3, H, d)
    spread = eigenvalues[..., -1] - eigenvalues[..., 0]                                      # (3, H)
    assert bool((spread <= math.log(1.5) + 1e-5).all())                # cond_2(e^X) <= kappa_max per head
    assert int(counter[0]) > 0                                         # the bound acted on touched rows


def test_row_counts_round_trip_through_a_checkpoint(tmp_path: Path) -> None:
    tokens, targets = _batch()
    cfg = _lazy_cfg()
    torch.manual_seed(0)
    model = VFEModel(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_lambda(s, cfg))
    for _ in range(2):
        train_step(model, optimizer, scheduler, tokens, targets, grad_clip=1.0)
    artifacts = RunArtifacts(tmp_path / "r", cfg, model)
    path = artifacts.save_checkpoint(2, model, optimizer, cfg)

    torch.manual_seed(1)
    fresh = VFEModel(cfg)
    fresh_optimizer = build_optimizer(fresh, cfg)
    assert load_checkpoint(path, fresh, fresh_optimizer, cfg=cfg) == 2
    phi, fresh_phi = model.prior_bank.phi_embed, fresh.prior_bank.phi_embed
    saved, loaded = optimizer.state[phi], fresh_optimizer.state[fresh_phi]
    assert set(loaded) == {"step", "exp_avg", "exp_avg_sq", "row_step"}
    for key in ("exp_avg", "exp_avg_sq", "row_step"):
        assert torch.equal(loaded[key], saved[key]), key
    assert float(loaded["step"]) == float(saved["step"]) == 2.0
    assert torch.equal(fresh_phi.detach(), phi.detach())
    # The continuation steps both identically (at the scheduled rate, which a resumed run's scheduler restores).
    for live, restored in zip(optimizer.param_groups, fresh_optimizer.param_groups):
        restored["lr"] = live["lr"]
    g = torch.Generator().manual_seed(3)
    grad = torch.randn(phi.shape, generator=g, dtype=phi.dtype)
    grad[0] = 0.0
    for opt, p in ((optimizer, phi), (fresh_optimizer, fresh_phi)):
        opt.zero_grad(set_to_none=True)
        p.grad = grad.clone()
        opt.table_statistics_provider.set_batch(tokens, torch.tensor(11), targets=targets)
        opt.step()
        opt.table_statistics_provider.set_batch(None, None)
    assert torch.equal(fresh_phi.detach(), phi.detach())


def test_dense_checkpoint_resumes_into_the_row_lazy_optimizer() -> None:
    w = torch.nn.Parameter(torch.ones(3, 2, dtype=torch.float64))
    dense = torch.optim.AdamW([{"params": [w], "lr": 0.1, "weight_decay": 0.0}], foreach=False)
    w.grad = torch.ones_like(w)
    dense.step()
    lazy = RowLazyAdamW([{"params": [w], "lr": 0.1, "weight_decay": 0.0, "sparse_rows": True}])
    saved = dense.state_dict()
    saved["param_groups"][0]["sparse_rows"] = True       # load_checkpoint reapplies the live group metadata
    lazy.load_state_dict(saved)
    grad = torch.zeros_like(w)
    grad[0] = 1.0
    w.grad = grad
    lazy.step()
    assert lazy.state[w]["row_step"].tolist() == [2.0, 1.0, 1.0]           # resumed from the dense count


# ---------------------------------------------------------------------------------------------------------------------
# m_decode_update_mode: natural-gradient step of the untied decode bank
# ---------------------------------------------------------------------------------------------------------------------

def _decode_cfg(**overrides: object):
    return test_config(**{**dict(vocab_size=9, embed_dim=4, n_heads=2, max_seq_len=6, n_layers=1,
                                 decode_mode="diagonal_chunked", untie_decode_bank=True,
                                 m_decode_update_mode="fisher"), **overrides})


def _decode_tables(model: VFEModel):
    return model.prior_bank.decode_mu_embed, model.prior_bank.decode_sigma_log_embed


def _fill_grads(model: VFEModel, seed: int = 0) -> None:
    g = torch.Generator().manual_seed(seed)
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.randn(p.shape, generator=g, dtype=p.dtype)


def _target_counts(targets: torch.Tensor, V: int) -> torch.Tensor:
    valid = targets[targets != -100]
    return torch.bincount(valid.reshape(-1), minlength=V).double()


def test_default_decode_mode_keeps_plain_adamw_groups() -> None:
    cfg = _decode_cfg(m_decode_update_mode="adamw")
    optimizer = build_optimizer(VFEModel(cfg), cfg)
    assert type(optimizer) is torch.optim.AdamW
    assert not any(group.get("table_fisher") for group in optimizer.param_groups)
    assert "m_table_fisher_lr" in cfg.inactive_fields()
    assert "m_table_fisher_lr" not in _decode_cfg().inactive_fields()
    assert "m_decode_update_mode" in test_config().inactive_fields()        # linear decode


def test_decode_step_equals_the_closed_form_natural_gradient() -> None:
    cfg = _decode_cfg(m_table_fisher_lr=0.3, m_table_fisher_trust_radius=1e9, decode_mu_weight_decay=0.2,
                      sigma_weight_decay=0.05)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    optimizer = build_optimizer(model, cfg)
    assert isinstance(optimizer, TableFisherAdamW)
    m, log_t = _decode_tables(model)
    with torch.no_grad():
        log_t.add_(torch.linspace(-0.5, 0.5, log_t.numel()).view_as(log_t))   # a non-uniform metric point
    _fill_grads(model)
    tokens, targets = _batch()
    count = _target_counts(targets, cfg.vocab_size)
    scored = torch.tensor(int((targets != -100).sum()))
    N = float(scored)
    t = log_t.detach().double().exp().clamp(min=model.prior_bank.eps)
    weight = (N / count.clamp_min(1.0)).unsqueeze(-1)
    groups = {g.get("table_fisher"): g for g in optimizer.param_groups if g.get("table_fisher")}
    assert set(groups) == {"decode_mu", "decode_sigma"}
    expected = {}
    for name, p, direction in (("decode_mu", m, t * m.grad.double()), ("decode_sigma", log_t, 2.0 * log_t.grad.double())):
        decay = 1.0 - groups[name]["lr"] * groups[name]["weight_decay"]
        expected[name] = p.detach().double() * decay - 0.3 * weight * direction
    others = {id(p): p.detach().clone() for p in model.parameters() if p is not m and p is not log_t}
    optimizer.table_statistics_provider.set_batch(tokens, scored, targets=targets)
    optimizer.step()
    torch.testing.assert_close(m.detach().double(), expected["decode_mu"], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(log_t.detach().double(), expected["decode_sigma"], rtol=1e-6, atol=1e-7)
    assert m.grad is None and log_t.grad is None                          # consumed by the Fisher step
    assert any(not torch.equal(p.detach(), others[id(p)]) for p in model.parameters()
               if id(p) in others)                                        # AdamW stepped the rest


def test_decode_trust_radius_bounds_each_row_and_the_schedule_scales_it() -> None:
    cfg = _decode_cfg(m_table_fisher_lr=10.0, m_table_fisher_trust_radius=0.05, decode_mu_weight_decay=0.0,
                      sigma_weight_decay=0.0)
    tokens, targets = _batch()
    scored = torch.tensor(int((targets != -100).sum()))
    norms = {}
    for factor in (1.0, 0.5):
        torch.manual_seed(0)
        model = VFEModel(cfg)
        optimizer = build_optimizer(model, cfg)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
            group["lr"] = group["lr"] * factor
        m, log_t = _decode_tables(model)
        _fill_grads(model)
        m0, l0 = m.detach().double().clone(), log_t.detach().double().clone()
        t = l0.exp().clamp(min=model.prior_bank.eps)
        optimizer.table_statistics_provider.set_batch(tokens, scored, targets=targets)
        optimizer.step()
        dm, dl = m.detach().double() - m0, log_t.detach().double() - l0
        norms[factor] = ((dm.square() / t).sum(-1) + 0.5 * dl.square().sum(-1)).sqrt()      # (V,)
    torch.testing.assert_close(norms[1.0], torch.full_like(norms[1.0], 0.05), rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(norms[0.5], 0.5 * norms[1.0], rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("overrides", [dict(decode_mode="linear"), dict(untie_decode_bank=False)])
def test_decode_validator_refuses_banks_the_metric_does_not_describe(overrides: dict) -> None:
    with pytest.raises(ValueError, match="m_decode_update_mode"):
        _decode_cfg(**overrides)


@pytest.mark.parametrize("extra", [dict(), dict(m_table_update_mode="fisher"), dict(m_phi_sparse_rows=True)])
def test_train_step_with_the_decode_fisher_step(extra: dict) -> None:
    tokens, targets = _batch()
    torch.manual_seed(0)
    cfg = _decode_cfg(warmup_steps=1, max_steps=4, **extra)
    model = VFEModel(cfg)
    optimizer = build_optimizer(model, cfg)
    if extra.get("m_phi_sparse_rows"):
        assert isinstance(optimizer, RowLazyTableFisherAdamW)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_lambda(s, cfg))
    m, log_t = _decode_tables(model)
    before = (m.detach().clone(), log_t.detach().clone())
    for _ in range(2):
        loss = train_step(model, optimizer, scheduler, tokens, targets, grad_clip=1.0)
        assert math.isfinite(loss)
    assert not torch.equal(m.detach(), before[0])
    assert not torch.equal(log_t.detach(), before[1])
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert optimizer.table_statistics_provider._targets is None            # cleared after the step


def test_statistics_require_targets_for_the_decode_step() -> None:
    cfg = _decode_cfg()
    model = VFEModel(cfg)
    optimizer = build_optimizer(model, cfg)
    _fill_grads(model)
    tokens, _ = _batch()
    optimizer.table_statistics_provider.set_batch(tokens, torch.tensor(11))
    with pytest.raises(RuntimeError, match="targets"):
        optimizer.step()
    assert isinstance(optimizer.table_statistics_provider, TableFisherStatistics)
