"""Training loop and optimizer fixes of the October 10 round-2 audit (punch rows 1, 8, 9, 10, 20, 34-37, 43, 57, 58).

Tiny CPU models only (K=4, two heads, N<=6, vocab <= 9). Covers: the per-role clip is unchanged under
grad_clip_scope='role' and one bound per group under 'group', with the realized coefficients reported (#8); groups
marked table_fisher are never clipped (#9); a failed first log-row diagnostic no longer latches the short key set,
and the diagnostics run on the precision ladder (#10); the table rates are marked inert under the Fisher step when
nothing else reads them and the realized table step size is logged (#20, #37); the base step runs before the table
step, so the frame metric reads pre-step rows (#34); the table counts and the device overflow gate keep their values
(#57); the interval statistics and spike rows of the divergence monitor (#1); and the replay of a step skipped
for non-finite gradients (#58).
"""
import csv
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
import torch
from torch.utils.data import DataLoader

import vfe3.run_artifacts as ra
import vfe3.train as T
from tests.profiles import test_config
from vfe3.config import VFE3Config
from vfe3.data.datasets import TokenWindows
from vfe3.model.model import VFEModel
from vfe3.table_optim import TABLE_FISHER_DIAGNOSTIC_KEYS, TableFisherGaugeManifoldAdamW, type_counts

B, N, V = 2, 4, 7
CPU = torch.device("cpu")


@pytest.fixture(autouse=True)
def _no_figure_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(T, "_save_eval_attention_maps", lambda *args, **kwargs: None)
    monkeypatch.setattr(ra, "_run_figures_isolated", lambda *args, **kwargs: True)
    torch.set_num_threads(1)


def _cfg(**overrides: Any) -> VFE3Config:
    values: Dict[str, Any] = dict(
        vocab_size=V, embed_dim=4, n_heads=2, batch_size=B, max_seq_len=N, eval_stride=2,
        n_layers=1, n_e_steps=1, warmup_steps=0, max_steps=6, log_interval=0, eval_interval=0,
        checkpoint_interval=0, skip_belief_sigma_update=True, generate_embedding_figures=False,
        pos_phi="learned", use_block_mlp=True, block_mlp_expansion=1,
        beta_attention_prior="causal_alibi", family="gaussian_diagonal", gauge_group="block_glk",
        attention_mode="projected_predictive", attention_residual=True,
        norm_type_block="layernorm", layernorm_affine=True, decode_mode="linear")
    values.update(overrides)
    return test_config(**values)


def _fisher_cfg(**overrides: Any) -> VFE3Config:
    return test_config(**{**dict(vocab_size=9, embed_dim=4, n_heads=2, max_seq_len=6, n_layers=1,
                                 decode_mode="linear", m_table_update_mode="fisher"), **overrides})


def _loaders(cfg: VFE3Config) -> Tuple[DataLoader, DataLoader]:
    train_ds = TokenWindows(torch.arange(41) % V, N)
    val_ds = TokenWindows((torch.arange(23) * 3) % V, N, pad_final=True, stride=cfg.eval_stride)
    return (DataLoader(train_ds, batch_size=B, shuffle=False, drop_last=True),
            DataLoader(val_ds, batch_size=2, shuffle=False))


def _model(cfg: VFE3Config, seed: int = 5) -> VFEModel:
    torch.manual_seed(seed)
    return VFEModel(cfg)


def _train(model: VFEModel, cfg: VFE3Config, loaders: Tuple[DataLoader, DataLoader], *,
           artifacts: Any = None, log_interval: int = 0, n_steps: int = 6,
           logger: logging.Logger = logging.getLogger("audit2-1011")) -> List[float]:
    torch.manual_seed(11)
    return T.train(model, loaders[0], cfg, n_steps=n_steps, grad_clip=cfg.grad_clip,
                   log_interval=log_interval, eval_interval=0, val_loader=loaders[1], device=CPU,
                   logger=logger, artifacts=artifacts, generate_samples=False)


def _rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _num(value: str) -> float:
    return float("nan") if value == "" else float(value)


def _batch(cfg: VFE3Config, seed: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return (torch.randint(0, cfg.vocab_size, (B, cfg.max_seq_len), generator=g),
            torch.randint(0, cfg.vocab_size, (B, cfg.max_seq_len), generator=g))


def _step_grads(cfg: VFE3Config, grad_clip: Any, *, scope: str = "role", status: Any = None,
                reference: bool = False, monkeypatch: Any = None) -> Tuple[Dict[str, torch.Tensor], Any]:
    r"""Gradients that reach optimizer.step for one train_step from a fixed model and batch."""
    model = _model(cfg)
    optimizer = T.build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    names = {id(p): n for n, p in model.named_parameters()}
    seen: Dict[str, torch.Tensor] = {}
    optimizer.register_step_pre_hook(lambda *_: seen.update(
        {names[id(p)]: p.grad.clone() for g in optimizer.param_groups for p in g["params"] if p.grad is not None}))
    if reference:
        # The former clip: one clip_grad_norm_ per nonempty role after the host barrier.
        monkeypatch.setattr(torch.nn.utils, "clip_grads_with_norm_",
                            lambda parameters, max_norm, total_norm, **_: torch.nn.utils.clip_grad_norm_(
                                parameters, max_norm))
    tokens, targets = _batch(cfg)
    T.train_step(model, optimizer, scheduler, tokens, targets, grad_clip=grad_clip, grad_clip_scope=scope,
                 status_out=status)
    if reference:
        monkeypatch.undo()
    return seen, optimizer


# ------------------------------------------------------------------------- #8, #9: clip sets and coefficients

def test_role_scope_clips_bitwise_as_the_former_per_role_clip(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg()
    status: Dict[str, Any] = {}
    new, _ = _step_grads(cfg, 1e-3, status=status)
    ref, _ = _step_grads(cfg, 1e-3, reference=True, monkeypatch=monkeypatch)
    silent, _ = _step_grads(cfg, 1e-3)                              # no status_out: the former call
    assert new and new.keys() == ref.keys() == silent.keys()
    for name in ref:
        assert torch.equal(new[name], ref[name]), name
        assert torch.equal(new[name], silent[name]), name
    raw, _ = _step_grads(cfg, None)
    by_role: Dict[str, float] = {}
    model = _model(cfg)
    optimizer = T.build_optimizer(model, cfg)
    names = {id(p): n for n, p in model.named_parameters()}
    for group in optimizer.param_groups:
        sq = sum(float(raw[names[id(p)]].double().square().sum()) for p in group["params"] if names[id(p)] in raw)
        by_role[group["role"]] = by_role.get(group["role"], 0.0) + sq
    for role, sq in by_role.items():
        norm = math.sqrt(sq)
        assert status["role_grad_norms"][role] == pytest.approx(norm, rel=1e-5)
        assert status["clip_coefs"][role] == pytest.approx(min(1.0, 1e-3 / (norm + 1e-6)), rel=1e-5)
    assert status["did_step"] and not status["nonfinite_skip"] and math.isfinite(status["ce"])


def test_group_scope_bounds_each_optimizer_group_separately() -> None:
    cfg = _cfg(grad_clip_scope="group")
    clip = 1e-3
    status: Dict[str, Any] = {}
    clipped, _ = _step_grads(cfg, clip, scope="group", status=status)
    raw, _ = _step_grads(cfg, None)
    model = _model(cfg)
    optimizer = T.build_optimizer(model, cfg)
    names = {id(p): n for n, p in model.named_parameters()}
    coefficients: Dict[str, float] = {}
    for group in optimizer.param_groups:
        members = [names[id(p)] for p in group["params"] if names[id(p)] in raw]
        if not members:
            continue
        norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(raw[n]) for n in members]))
        coefficient = torch.clamp(clip / (norm + 1e-6), max=1.0)
        for n in members:
            torch.testing.assert_close(clipped[n], raw[n] * coefficient, rtol=1e-6, atol=0.0)
        after = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(clipped[n]) for n in members]))
        assert float(after) <= clip * (1 + 1e-5)
        coefficients[group["role"]] = min(coefficients.get(group["role"], 1.0), float(coefficient))
    for role, coefficient in coefficients.items():
        assert status["clip_coefs"][role] == pytest.approx(coefficient, rel=1e-5)
    # Several mu-role groups: under 'role' their union is bounded once, so the scopes differ.
    role_clipped, _ = _step_grads(cfg, clip, scope="role")
    assert any(not torch.equal(role_clipped[n], clipped[n]) for n in clipped)


def test_table_fisher_groups_are_never_clipped() -> None:
    cfg = _fisher_cfg()
    status: Dict[str, Any] = {}
    clipped, optimizer = _step_grads(cfg, 1e-6, status=status)
    raw, _ = _step_grads(cfg, None)
    for name in ("prior_bank.mu_embed", "prior_bank.sigma_log_embed"):
        assert torch.equal(clipped[name], raw[name]), name               # the table gradient is unscaled
    others = [n for n in raw if n not in ("prior_bank.mu_embed", "prior_bank.sigma_log_embed")]
    assert others and all(not torch.equal(clipped[n], raw[n]) for n in others if raw[n].abs().sum() > 0)
    table = {"mu": torch.linalg.vector_norm(raw["prior_bank.mu_embed"]),
             "sigma": torch.linalg.vector_norm(raw["prior_bank.sigma_log_embed"])}
    assert status["role_grad_norms"]["sigma"] >= float(table["sigma"]) * (1 - 1e-6)   # monitored, not clipped
    assert set(status["clip_coefs"]) <= {"mu", "sigma", "phi"}


def test_clip_scope_config_is_validated_and_inert_without_a_clip() -> None:
    with pytest.raises(ValueError, match="grad_clip_scope"):
        _cfg(grad_clip_scope="layer")
    assert "grad_clip_scope" in _cfg(grad_clip=0.0).inactive_fields()
    assert "grad_clip_scope" not in _cfg().inactive_fields()


# ------------------------------------------------------------------------- #20, #37: table rates and diagnostics

def _inert_table_rate_cfg(**overrides: Any) -> VFE3Config:
    return _fisher_cfg(**{**dict(weight_decay=0.0, mu_weight_decay=0.0, sigma_weight_decay=0.0,
                                 m_decode_mu_lr=0.01, m_block_mlp_lr=0.01, m_relational_score_lr=0.01,
                                 m_relational_content_lr=0.01, m_layernorm_lr=0.01), **overrides})


def test_table_rates_are_inert_under_fisher_only_when_nothing_reads_them() -> None:
    inactive = _inert_table_rate_cfg().inactive_fields()
    assert {"m_p_mu_lr", "m_p_sigma_lr"} <= set(inactive)
    for overrides in (dict(m_block_mlp_lr=None), dict(m_decode_mu_lr=None), dict(m_relational_score_lr=None),
                      dict(m_layernorm_lr=None), dict(mu_weight_decay=0.01), dict(mu_weight_decay=None,
                                                                                      weight_decay=0.01)):
        assert "m_p_mu_lr" not in _inert_table_rate_cfg(**overrides).inactive_fields(), overrides
    assert "m_p_sigma_lr" not in _inert_table_rate_cfg(sigma_weight_decay=0.01).inactive_fields()
    untied = _inert_table_rate_cfg(decode_mode="diagonal_chunked", m_decode_sigma_lr=None)
    assert "m_p_sigma_lr" not in untied.inactive_fields() and "m_p_mu_lr" in untied.inactive_fields()
    assert not {"m_p_mu_lr", "m_p_sigma_lr"} & set(_cfg(weight_decay=0.0, mu_weight_decay=0.0,
                                                        sigma_weight_decay=0.0).inactive_fields())
    assert "fisher eta=0.1" in T._fmt_table_lrs(_fisher_cfg()) and "mu=0.025" in T._fmt_table_lrs(_cfg())


def test_table_rate_column_and_step_norm_report_the_realized_scheduled_step() -> None:
    cfg = _fisher_cfg(m_table_fisher_lr=10.0, m_table_fisher_trust_radius=0.05, weight_decay=0.0,
                      mu_weight_decay=0.0, sigma_weight_decay=0.0)
    model = _model(cfg)
    optimizer = T.build_optimizer(model, cfg)
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
        group["lr"] = 0.5 * group["lr"]                               # schedule factor 1/2
    assert T._table_fisher_learning_rates(optimizer) == {"mu": 5.0, "sigma": 5.0}
    assert T._table_fisher_learning_rates(T.build_optimizer(_model(_cfg()), _cfg())) == {}
    g = torch.Generator().manual_seed(0)
    for p in model.parameters():
        p.grad = torch.randn(p.shape, generator=g)
    optimizer.table_statistics_provider.set_batch(torch.tensor([[1, 2, 3, 3, 4, 1]]), torch.tensor(6))
    optimizer._collect_table_diag = True
    optimizer._table_fisher_step()
    diag = optimizer._table_diag
    assert diag["table_fisher_active_rows"] == 4.0
    assert diag["table_fisher_step_norm_mean"] == pytest.approx(0.5 * 0.05, rel=1e-5)   # factor * r, not r
    assert diag["table_fisher_trust_scale_min"] < 1.0


def test_skipped_log_step_does_not_report_the_previous_table_diagnostics(tmp_path: Path,
                                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _fisher_cfg(vocab_size=V, max_seq_len=N, warmup_steps=0, log_interval=1, eval_interval=0,
                      checkpoint_interval=0)
    model = _model(cfg)
    real_forward = model.forward
    calls = {"n": 0}

    def forward(*args: Any, **kwargs: Any) -> Any:
        out = real_forward(*args, **kwargs)
        if kwargs.get("return_decode_stats"):
            calls["n"] += 1
            if calls["n"] == 2:                                         # the second train step: a NaN loss
                return (out[0], out[1] * float("nan"), *out[2:])
        return out

    monkeypatch.setattr(model, "forward", forward)
    artifacts = ra.RunArtifacts(tmp_path / "run", cfg, model, save_model_artifacts=False)
    _train(model, cfg, _loaders(cfg), artifacts=artifacts, log_interval=1, n_steps=3)
    rows = _rows(tmp_path / "run" / "metrics.csv")
    assert [row["step_skipped"] for row in rows] == ["0.0", "1.0", "0.0"]
    assert _num(rows[0]["table_fisher_active_rows"]) > 0
    assert all(rows[1][key] == "" for key in TABLE_FISHER_DIAGNOSTIC_KEYS)
    assert float(rows[0]["lr_mu"]) == pytest.approx(cfg.m_table_fisher_lr)     # eta * lr_t/lr_0 at warmup 0


# ------------------------------------------------------------------------- #34, #57: table step order and syncs

def test_frame_metric_reads_the_pre_step_prior_rows() -> None:
    cfg = _fisher_cfg(m_phi_update_mode="fisher_group", m_table_fisher_trust_radius=1e9, weight_decay=0.0,
                      mu_weight_decay=0.0, phi_weight_decay=0.0)
    model = _model(cfg)
    optimizer = T.build_optimizer(model, cfg)
    assert isinstance(optimizer, TableFisherGaugeManifoldAdamW)
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
    _, loss, _ = model(tokens[:, :-1], tokens[:, 1:])
    loss.backward()
    before = (model.prior_bank.mu_embed.detach().clone(), model.prior_bank.sigma_log_embed.detach().clone())
    provider = optimizer.phi_statistics_provider
    seen: List[Tuple[torch.Tensor, torch.Tensor]] = []

    class Recording:
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            seen.append((model.prior_bank.mu_embed.detach().clone(),
                         model.prior_bank.sigma_log_embed.detach().clone()))
            return provider(*args, **kwargs)

    provider.set_batch(tokens[:, :-1], torch.tensor(5))
    optimizer.phi_statistics_provider = Recording()
    optimizer.table_statistics_provider.set_batch(tokens[:, :-1], torch.tensor(5))
    optimizer.step()
    assert seen
    for mu, log_s in seen:
        assert torch.equal(mu, before[0]) and torch.equal(log_s, before[1])
    assert not torch.equal(model.prior_bank.mu_embed.detach(), before[0])     # the table step still ran
    assert model.prior_bank.mu_embed.grad is None


def test_type_counts_equal_bincount() -> None:
    g = torch.Generator().manual_seed(4)
    for shape, vocab in (((3, 5), 9), ((1, 1), 2), ((4, 7), 50)):
        tokens = torch.randint(0, vocab, shape, generator=g)
        expected = torch.bincount(tokens.reshape(-1), minlength=vocab)[:vocab]
        counts = type_counts(tokens, vocab, CPU)
        assert counts.dtype == torch.int64 and torch.equal(counts, expected)


def test_device_overflow_gate_keeps_the_values() -> None:
    cfg = _fisher_cfg(m_table_fisher_trust_radius=0.05)
    results = []
    for found_inf in (None, torch.zeros(()), torch.ones(())):
        model = _model(cfg)
        optimizer = T.build_optimizer(model, cfg)
        g = torch.Generator().manual_seed(0)
        for p in model.parameters():
            p.grad = torch.randn(p.shape, generator=g)
        if found_inf is not None:
            optimizer.found_inf = found_inf
        optimizer.table_statistics_provider.set_batch(torch.tensor([[1, 2, 3, 3, 4, 1]]), torch.tensor(6))
        optimizer._collect_table_diag = True
        before = model.prior_bank.mu_embed.detach().clone()
        optimizer._table_fisher_step()
        results.append((before, model.prior_bank.mu_embed.detach().clone(),
                        model.prior_bank.sigma_log_embed.detach().clone(), dict(optimizer._table_diag)))
        assert model.prior_bank.mu_embed.grad is None
    (b0, mu0, l0, d0), (_, mu1, l1, d1), (b2, mu2, _, d2) = results
    assert torch.equal(mu0, mu1) and torch.equal(l0, l1) and d0 == d1           # finite: values unchanged
    assert not torch.equal(mu0, b0)
    assert torch.equal(mu2, b2) and d2 == {}                                    # overflow: no step, no diagnostics


def test_variances_equal_the_detached_values() -> None:
    from vfe3.numerics import bounded_variance_from_log
    model = _model(_fisher_cfg())
    optimizer = T.build_optimizer(model, _fisher_cfg())
    pb = model.prior_bank
    s = optimizer.table_statistics_provider.variances(torch.float32)
    assert not s.requires_grad
    assert torch.equal(s, bounded_variance_from_log(pb.sigma_log_embed.detach(), eps=pb.eps))


# ------------------------------------------------------------------------- #10: the diagnostics latch and ladder

def _flaky_diagnostics(model: VFEModel, monkeypatch: pytest.MonkeyPatch, fail: Any) -> Dict[str, int]:
    original = model.diagnostics
    calls = {"n": 0}

    def flaky(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if fail(calls["n"]):
            raise RuntimeError("synthetic log-step diagnostic failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "diagnostics", flaky)
    return calls


def test_failed_first_row_diagnostic_keeps_the_full_column_set(tmp_path: Path,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg()
    healthy = _model(cfg)
    reference = ra.RunArtifacts(tmp_path / "healthy", cfg, healthy, save_model_artifacts=False)
    _train(healthy, cfg, _loaders(cfg), artifacts=reference, log_interval=2)
    model = _model(cfg)
    _flaky_diagnostics(model, monkeypatch, lambda n: n == 1)
    artifacts = ra.RunArtifacts(tmp_path / "run", cfg, model, save_model_artifacts=False)
    _train(model, cfg, _loaders(cfg), artifacts=artifacts, log_interval=2)
    rows = _rows(tmp_path / "run" / "metrics.csv")
    healthy_rows = _rows(tmp_path / "healthy" / "metrics.csv")
    assert list(rows[0]) == list(healthy_rows[0])                 # the full set, not the nine fallback keys
    relational = [key for key in rows[0] if key.startswith(("relational_", "qkv_"))]
    assert relational and all(rows[0][key] == "" for key in relational)
    assert [int(row["step"]) for row in rows] == [2, 4, 6]
    assert all(any(row[key] != "" for key in relational) for row in rows[1:])
    assert artifacts.diagnostics_failures == {"training_diagnostics": 1}


def test_diagnostics_replay_at_a_stricter_matmul_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(fp32_matmul_precision="high")
    model = _model(cfg)
    policies: List[str] = []
    original = model.diagnostics

    def strict(*args: Any, **kwargs: Any) -> Any:
        policies.append(torch.get_float32_matmul_precision())
        if policies[-1] != "highest":
            raise ValueError("projected covariance is not positive definite")
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "diagnostics", strict)
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        artifacts = ra.RunArtifacts(tmp_path / "run", cfg, model, save_model_artifacts=False)
        _train(model, cfg, _loaders(cfg), artifacts=artifacts, log_interval=3)
    finally:
        torch.set_float32_matmul_precision(previous)
    assert policies == ["high", "highest"] * 2
    assert artifacts.diagnostics_failures == {}
    rows = _rows(tmp_path / "run" / "metrics.csv")
    assert all(math.isfinite(_num(row["attn_entropy"])) for row in rows)


# ------------------------------------------------------------------------- #1: interval statistics and spikes

def test_rows_carry_interval_statistics_of_the_steps_since_the_previous_row(tmp_path: Path) -> None:
    cfg = _cfg(grad_clip=1e-3)
    model = _model(cfg)
    artifacts = ra.RunArtifacts(tmp_path / "run", cfg, model, save_model_artifacts=False)
    losses = _train(model, cfg, _loaders(cfg), artifacts=artifacts, log_interval=3)
    rows = _rows(tmp_path / "run" / "metrics.csv")
    assert [int(row["step"]) for row in rows] == [3, 6]
    for row, window in zip(rows, (losses[0:3], losses[3:6])):
        assert all(key in row for key in T.DIVERGENCE_COLUMNS)
        assert float(row["interval_steps"]) == 3.0
        assert float(row["train_loss_interval_mean"]) == pytest.approx(sum(window) / 3, rel=1e-12)
        assert float(row["train_loss_interval_max"]) == pytest.approx(max(window), rel=1e-12)
        assert float(row["grad_norm_mu_interval_max"]) >= float(row["grad_norm_mu_interval_mean"]) > 0.0
        assert float(row["grad_clip_binding_frac"]) == 1.0          # c = 1e-3 binds on every step
        assert 0.0 < float(row["grad_clip_coef_mu"]) < 1.0
        assert float(row["divergence_spike"]) == 0.0
        assert math.isfinite(float(row["train_ce_interval_mean"]))


def test_a_spike_warns_and_forces_a_metrics_row(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setattr(T, "DIVERGENCE_EMA_WARMUP", 2)
    real_step = T.train_step
    calls = {"n": 0}

    def spiking(*args: Any, **kwargs: Any) -> float:
        loss = real_step(*args, **kwargs)
        calls["n"] += 1
        if calls["n"] == 5:                                          # step 5: the mu pre-clip norm jumps 1000x
            kwargs["status_out"]["role_grad_norms"]["mu"] *= 1e3
        return loss

    monkeypatch.setattr(T, "train_step", spiking)
    cfg = _cfg(max_steps=8)
    model = _model(cfg)
    artifacts = ra.RunArtifacts(tmp_path / "run", cfg, model, save_model_artifacts=False)
    with caplog.at_level(logging.WARNING, logger="audit2-1011"):
        _train(model, cfg, _loaders(cfg), artifacts=artifacts, log_interval=4, n_steps=8)
    assert any("divergence spike" in record.getMessage() and "grad_norm_mu" in record.getMessage()
               for record in caplog.records)
    rows = _rows(tmp_path / "run" / "metrics.csv")
    assert [int(row["step"]) for row in rows] == [4, 5, 8]
    forced = rows[1]
    assert float(forced["divergence_spike"]) == 1.0 and forced["attn_entropy"] == ""
    assert float(forced["grad_norm_mu"]) > 100.0 * float(rows[0]["grad_norm_mu"])
    assert float(rows[2]["interval_steps"]) == 4.0                  # a forced row does not end the interval
    assert float(rows[0]["divergence_spike"]) == float(rows[2]["divergence_spike"]) == 0.0


def _nan_after(model: VFEModel, monkeypatch: pytest.MonkeyPatch, healthy_steps: int) -> None:
    real_forward = model.forward
    calls = {"n": 0}

    def forward(*args: Any, **kwargs: Any) -> Any:
        out = real_forward(*args, **kwargs)
        if kwargs.get("return_decode_stats"):
            calls["n"] += 1
            if calls["n"] > healthy_steps:
                return (out[0], out[1] * float("nan"), *out[2:])
        return out

    monkeypatch.setattr(model, "forward", forward)


def test_rejected_updates_never_stop_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(max_steps=8)
    model = _model(cfg)
    _nan_after(model, monkeypatch, healthy_steps=2)
    artifacts = ra.RunArtifacts(tmp_path / "run", cfg, model, save_model_artifacts=False)
    losses = _train(model, cfg, _loaders(cfg), artifacts=artifacts, n_steps=8)
    assert len(losses) == 8 and artifacts.final_step == 8
    realized = artifacts.realized_updates
    assert realized["attempted_steps"] == 8 and realized["accepted_updates"] == 2
    assert "training_status" not in realized


# ------------------------------------------------------------------------- #58: replay of a non-finite skip

def test_nonfinite_skip_under_a_reduced_policy_is_replayed_stricter(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(fp32_matmul_precision="high")
    model = _model(cfg)
    reference = _model(cfg)
    real_forward = model.forward

    def forward(*args: Any, **kwargs: Any) -> Any:
        out = real_forward(*args, **kwargs)
        if kwargs.get("return_decode_stats") and torch.get_float32_matmul_precision() != "highest":
            return (out[0], out[1] * float("nan"), *out[2:])
        return out

    monkeypatch.setattr(model, "forward", forward)
    tokens, targets = _batch(cfg)
    previous = torch.get_float32_matmul_precision()
    records: List[Dict[str, object]] = []
    status: Dict[str, Any] = {}
    try:
        torch.set_float32_matmul_precision("high")
        optimizer = T.build_optimizer(model, cfg)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
        torch.manual_seed(0)
        loss, replays = T._train_step_with_precision_retry(
            model, optimizer, scheduler, tokens, targets, logger=logging.getLogger("audit2-1011"), step=0,
            replay_log=records, grad_clip=1.0, status_out=status)
        torch.set_float32_matmul_precision("highest")
        ref_optimizer = T.build_optimizer(reference, cfg)
        ref_scheduler = torch.optim.lr_scheduler.LambdaLR(ref_optimizer, lambda step: 1.0)
        torch.manual_seed(0)
        ref_loss = T.train_step(reference, ref_optimizer, ref_scheduler, tokens, targets, grad_clip=1.0)
    finally:
        torch.set_float32_matmul_precision(previous)
    assert replays == 1 and status["did_step"] and math.isfinite(loss) and loss == ref_loss
    assert [(r["policy"], r["replay_policy"], r["exception"]) for r in records] == [
        ("high", "highest", "NonFiniteStepSkipped")]
    for (name, p), q in zip(model.named_parameters(), reference.parameters()):
        assert torch.equal(p.detach(), q.detach()), name


def test_a_skip_that_persists_at_highest_is_returned_as_a_skip(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(fp32_matmul_precision="high")
    model = _model(cfg)
    _nan_after(model, monkeypatch, healthy_steps=0)
    tokens, targets = _batch(cfg)
    previous = torch.get_float32_matmul_precision()
    status: Dict[str, Any] = {}
    records: List[Dict[str, object]] = []
    try:
        torch.set_float32_matmul_precision("high")
        optimizer = T.build_optimizer(model, cfg)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
        loss, replays = T._train_step_with_precision_retry(
            model, optimizer, scheduler, tokens, targets, logger=logging.getLogger("audit2-1011"), step=0,
            replay_log=records, grad_clip=1.0, status_out=status)
    finally:
        torch.set_float32_matmul_precision(previous)
    assert replays == 1 and not status["did_step"] and status["nonfinite_skip"] and not math.isfinite(loss)
    assert [r["exception"] for r in records] == ["NonFiniteStepSkipped"]
