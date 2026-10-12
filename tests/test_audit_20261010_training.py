"""Training-loop robustness and telemetry fixes of the October 10 deep audit (punch rows 4, 15, 16, 17, 29).

Tiny CPU models only (K=4, two heads, N=4, vocab 7; the spectrum probes use K <= 8). Covers: a failed
log-step diagnostic writes blank cells instead of aborting at the next metrics.csv write (#16); a
transient best_model.pt save failure is counted and retried instead of aborting (#17);
val_attn_collapsed_heads counts heads by their mean normalized entropy over the diagnostic windows
(#15); the self-loop spectral radius is computed on a bounded token subsample (#29); and train() and
evaluate() refuse a VFE model whose attention prior is not causal for next-token scoring (#4).
"""
from tests.profiles import test_config
import csv
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import vfe3.metrics as M
import vfe3.run_artifacts as ra
import vfe3.train as T
from vfe3.baseline import BaselineConfig, BaselineTransformer
from vfe3.config import VFE3Config
from vfe3.data.datasets import TokenWindows
from vfe3.model.model import VFEModel

B, N, V = 2, 4, 7


def _cfg(**overrides: Any) -> VFE3Config:
    values: Dict[str, Any] = dict(
        vocab_size=V, embed_dim=4, n_heads=2, batch_size=B, max_seq_len=N, eval_stride=2,
        n_layers=2, n_e_steps=1, warmup_steps=0, max_steps=6, log_interval=0, eval_interval=0,
        checkpoint_interval=0, skip_belief_sigma_update=True, generate_embedding_figures=False,
        pos_phi="learned", use_block_mlp=True, block_mlp_expansion=1,
        beta_attention_prior="causal_alibi", family="gaussian_diagonal", gauge_group="block_glk",
        attention_mode="projected_predictive", attention_residual=True,
        norm_type_block="layernorm", layernorm_affine=True, decode_mode="linear")
    values.update(overrides)
    return test_config(**values)


def _loaders(cfg: VFE3Config) -> Tuple[DataLoader, DataLoader]:
    train_ds = TokenWindows(torch.arange(41) % V, N)
    val_ds = TokenWindows((torch.arange(23) * 3) % V, N, pad_final=True, stride=cfg.eval_stride)
    return (DataLoader(train_ds, batch_size=B, shuffle=True),
            DataLoader(val_ds, batch_size=2, shuffle=False))


def _model(cfg: VFE3Config, seed: int = 5) -> VFEModel:
    torch.manual_seed(seed)
    return VFEModel(cfg)


def _train(model: VFEModel, cfg: VFE3Config, loaders: Tuple[DataLoader, DataLoader], *,
           artifacts: Any = None, log_interval: int = 0, eval_interval: int = 0) -> List[float]:
    torch.manual_seed(11)
    return T.train(model, loaders[0], cfg, n_steps=cfg.max_steps, grad_clip=1.0,
                   log_interval=log_interval, eval_interval=eval_interval, val_loader=loaders[1],
                   device=torch.device("cpu"), logger=logging.getLogger("audit-1010"),
                   artifacts=artifacts, generate_samples=False)


def _rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _num(value: str) -> float:
    return float("nan") if value == "" else float(value)


@pytest.fixture(autouse=True)
def _no_figure_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(T, "_save_eval_attention_maps", lambda *args, **kwargs: None)
    monkeypatch.setattr(ra, "_run_figures_isolated", lambda *args, **kwargs: True)
    torch.set_num_threads(1)


def _flaky_diagnostics(model: VFEModel, monkeypatch: pytest.MonkeyPatch, failing_call: int) -> Dict[str, int]:
    original = model.diagnostics
    calls = {"n": 0}

    def flaky(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == failing_call:
            raise RuntimeError("synthetic log-step diagnostic failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "diagnostics", flaky)
    return calls


# ------------------------------------------------------------------------- #16: a failed log-step diagnostic

def test_failed_log_step_diagnostic_writes_blank_cells_and_training_continues(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg()
    model = _model(cfg)
    calls = _flaky_diagnostics(model, monkeypatch, failing_call=2)
    run_dir = tmp_path / "run"
    artifacts = ra.RunArtifacts(run_dir, cfg, model, save_model_artifacts=False)
    losses = _train(model, cfg, _loaders(cfg), artifacts=artifacts, log_interval=2)
    assert len(losses) == cfg.max_steps and calls["n"] == 3
    assert artifacts.diagnostics_failures == {"training_diagnostics": 1}
    rows = _rows(run_dir / "metrics.csv")
    assert [int(row["step"]) for row in rows] == [2, 4, 6]
    assert [int(row["diagnostics_failures"]) for row in rows] == [0, 1, 1]
    diagnostic_columns = [key for key in rows[0] if key.startswith(("relational_", "qkv_"))]
    assert diagnostic_columns                                  # the first row established them
    for key in ("attn_entropy", "self_coupling", "inner_alignment_energy_total", *diagnostic_columns):
        assert rows[1][key] == "", key                        # the failed row: blank, not dropped
    for row in (rows[0], rows[2]):
        assert math.isfinite(_num(row["attn_entropy"]))
    assert any(rows[2][key] != "" for key in diagnostic_columns)


def test_failed_first_row_diagnostic_keeps_its_columns_for_the_run(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg()
    model = _model(cfg)
    _flaky_diagnostics(model, monkeypatch, failing_call=1)
    run_dir = tmp_path / "run"
    artifacts = ra.RunArtifacts(run_dir, cfg, model, save_model_artifacts=False)
    losses = _train(model, cfg, _loaders(cfg), artifacts=artifacts, log_interval=2)
    assert len(losses) == cfg.max_steps
    assert artifacts.diagnostics_failures == {"training_diagnostics": 1}
    rows = _rows(run_dir / "metrics.csv")
    assert [int(row["step"]) for row in rows] == [2, 4, 6]
    assert rows[0]["attn_entropy"] == ""
    assert all(math.isfinite(_num(row["attn_entropy"])) for row in rows[1:])


# ------------------------------------------------------------------------- #17: a transient best-model save failure

def _fail_first_best_save(monkeypatch: pytest.MonkeyPatch) -> Dict[str, int]:
    original = torch.save
    seen = {"failed": 0, "saved": 0}

    def flaky_save(obj: Any, path: Any, *args: Any, **kwargs: Any) -> Any:
        if Path(str(path)).name.startswith(".best_model.pt."):
            if seen["failed"] == 0:
                seen["failed"] += 1
                raise RuntimeError("[enforce fail at inline_container.cc] ios_base::badbit set")
            seen["saved"] += 1
        return original(obj, path, *args, **kwargs)

    monkeypatch.setattr(torch, "save", flaky_save)
    return seen


def test_failed_best_model_save_keeps_record_and_file_consistent(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg()
    model = _model(cfg)
    artifacts = ra.RunArtifacts(tmp_path / "run", cfg, model, dataset="tiny")
    _fail_first_best_save(monkeypatch)
    with pytest.raises(RuntimeError, match="badbit"):
        artifacts.maybe_save_best(2, model, 10.0)
    assert artifacts.best_val_ppl == float("inf") and artifacts.best_step is None
    assert not artifacts.best_path.exists()
    assert not list(artifacts.run_dir.glob(".best_model.pt.*"))      # the temporary was removed
    assert artifacts.maybe_save_best(4, model, 12.0) is True          # the next improving eval retries
    assert artifacts.best_val_ppl == 12.0 and artifacts.best_step == 4 and artifacts.best_path.is_file()


def test_transient_best_model_save_failure_does_not_abort_training(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    cfg = _cfg(max_steps=4)
    model = _model(cfg)
    artifacts = ra.RunArtifacts(tmp_path / "run", cfg, model, dataset="tiny")
    seen = _fail_first_best_save(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="audit-1010"):
        losses = _train(model, cfg, _loaders(cfg), artifacts=artifacts, eval_interval=2)
    assert len(losses) == cfg.max_steps
    assert seen == {"failed": 1, "saved": 1}
    assert artifacts.diagnostics_failures == {"best_model_save": 1}
    assert artifacts.best_step == 4 and math.isfinite(artifacts.best_val_ppl)
    assert artifacts.best_path.is_file()
    assert any("best_model.pt save at step 2 failed" in message for message in caplog.messages)
    rows = _rows(artifacts.run_dir / "metrics.csv")
    assert [int(row["diagnostics_failures"]) for row in rows] == [1, 1]


# ------------------------------------------------------------------------- #15: val_attn_collapsed_heads

def test_collapsed_head_count_uses_the_per_head_mean() -> None:
    rows = [{"entropy_norm": 0.30}, {"entropy_norm": 0.70}, {"entropy_norm": 0.4999},
            {"entropy_norm": float("nan")}]
    assert T._collapsed_head_count(rows) == 2.0
    assert math.isnan(T._collapsed_head_count([{"entropy_norm": float("nan")}]))
    assert math.isnan(T._collapsed_head_count([]))


def test_single_window_structure_no_longer_reports_the_head_count() -> None:
    beta = torch.tril(torch.ones(1, 2, 5, 5))
    beta = beta / beta.sum(dim=-1, keepdim=True)
    beta[0, 0, 3] = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0])         # one one-hot row in a uniform head
    out = T._attention_structure_metrics(beta, torch.tril(torch.ones(5, 5, dtype=torch.bool)))
    assert "val_attn_collapsed_heads" not in out
    assert out["val_attn_entropy_min_norm"] == pytest.approx(0.0, abs=1e-9)   # the worst row is still reported


def test_val_collapsed_heads_matches_the_diagnostic_window_attention_rows(tmp_path: Path) -> None:
    cfg = _cfg(max_steps=8)
    model = _model(cfg)
    run_dir = tmp_path / "run"
    artifacts = ra.RunArtifacts(run_dir, cfg, model, save_model_artifacts=False)
    _train(model, cfg, _loaders(cfg), artifacts=artifacts, log_interval=2, eval_interval=4)
    assert artifacts.diagnostics_failures == {}
    attention = _rows(run_dir / "diagnostics" / "attention_stats.csv")
    by_step: Dict[int, List[float]] = {}
    for row in attention:
        by_step.setdefault(int(row["step"]), []).append(_num(row["entropy_norm"]))
    assert sorted(by_step) == [4, 8]
    for row in _rows(run_dir / "metrics.csv"):
        step = int(row["step"])
        if step in by_step:
            expected = sum(value < M.COLLAPSE_ENTROPY_FRACTION for value in by_step[step])
            assert _num(row["val_attn_collapsed_heads"]) == expected, step
        else:
            assert row["val_attn_collapsed_heads"] == ""


# ------------------------------------------------------------------------- #29: self-loop spectral radius

def _random_maps(H: int, d: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    K = H * d
    return (torch.randn(H, d, K, generator=g, dtype=torch.float64),
            torch.randn(H, K, d, generator=g, dtype=torch.float64))


def _exhaustive_radii(T_hat: torch.Tensor, O_hat: torch.Tensor, beta_self: torch.Tensor,
                      residual: bool) -> np.ndarray:
    H, d, K = T_hat.shape
    W = np.einsum("hsk,gkt->hgst", T_hat.numpy(), O_hat.numpy()).transpose(0, 2, 1, 3).reshape(K, K)
    b = beta_self.double().numpy().reshape(-1, H, beta_self.shape[-1]).transpose(0, 2, 1).reshape(-1, H)
    radii = []
    for row in b:
        A = np.repeat(row, d)[:, None] * W
        radii.append(np.abs(np.linalg.eigvals(A) + (1.0 if residual else 0.0)).max())
    return np.asarray(radii)


@pytest.mark.parametrize("residual", [False, True])
def test_selfloop_radius_subsample_keeps_the_causal_maximum(residual: bool) -> None:
    H, d, Bs, Ns = 2, 2, 3, 100                                      # B N = 300 > 256 token positions
    T_hat, O_hat = _random_maps(H, d, seed=1)
    g = torch.Generator().manual_seed(2)
    beta_self = 0.9 * torch.rand(Bs, H, Ns, generator=g, dtype=torch.float64)
    beta_self[:, :, 0] = 1.0                                         # causal: position 0 routes only to itself
    got = M.content_link_spectrum(T_hat, O_hat, beta_self=beta_self, residual=residual)
    radii = _exhaustive_radii(T_hat, O_hat, beta_self, residual)
    sample = torch.linspace(0, radii.size - 1, M.SELFLOOP_RADIUS_SAMPLE, dtype=torch.float64).round().long()
    assert got["relational_link_selfloop_spectral_radius_max"] == pytest.approx(float(radii.max()), rel=1e-12)
    assert got["relational_link_selfloop_spectral_radius_mean"] == pytest.approx(
        float(radii[sample.numpy()].mean()), rel=1e-12)


def test_selfloop_radius_includes_unsampled_identity_tokens_and_is_a_lower_bound() -> None:
    H, d, Bs, Ns = 2, 2, 4, 100
    T_hat, O_hat = _random_maps(H, d, seed=3)
    g = torch.Generator().manual_seed(4)
    beta_self = 0.5 * torch.rand(Bs, H, Ns, generator=g, dtype=torch.float64)
    sample = set(torch.linspace(0, Bs * Ns - 1, M.SELFLOOP_RADIUS_SAMPLE,
                                dtype=torch.float64).round().long().tolist())
    outside = next(index for index in range(Bs * Ns) if index not in sample)
    beta_self[outside // Ns, :, outside % Ns] = 1.0                  # D_i = I at an unsampled token
    got = M.content_link_spectrum(T_hat, O_hat, beta_self=beta_self, residual=False)
    radii = _exhaustive_radii(T_hat, O_hat, beta_self, residual=False)
    assert got["relational_link_selfloop_spectral_radius_max"] == pytest.approx(float(radii[outside]), rel=1e-12)
    assert got["relational_link_selfloop_spectral_radius_max"] <= float(radii.max()) * (1 + 1e-12)


def test_selfloop_radius_without_identity_tokens_does_not_report_the_full_transfer() -> None:
    H, d, Bs, Ns = 2, 2, 3, 100
    T_hat, O_hat = _random_maps(H, d, seed=5)
    beta_self = torch.zeros(Bs, H, Ns, dtype=torch.float64)          # a no-self prior: beta_ii = 0
    got = M.content_link_spectrum(T_hat, O_hat, beta_self=beta_self, residual=True)
    assert got["relational_link_selfloop_spectral_radius_max"] == pytest.approx(1.0, abs=1e-12)
    assert got["relational_link_selfloop_spectral_radius_mean"] == pytest.approx(1.0, abs=1e-12)


# ------------------------------------------------------------------------- #4: non-causal priors refused

@pytest.mark.registry_mutation
def test_train_and_evaluate_refuse_a_non_causal_vfe_prior() -> None:
    r"""No built-in prior is non-causal since 2026-10-12; the guard reads the registry's causality metadata, so a
    newly registered prior without the causal declaration is refused."""
    from vfe3.attention_prior import _PRIOR_CAUSALITY, _PRIORS, register_prior

    name = "_test_bidirectional_zero"

    @register_prior(name)
    def _bidirectional(n_query: int, n_key: int, *, device: Any = None, dtype: torch.dtype = torch.float32,
                       **kwargs: Any) -> torch.Tensor:
        return torch.zeros(n_query, n_key, device=device, dtype=dtype)    # (Nq, Nk) sees the future

    try:
        cfg = _cfg(beta_attention_prior=name, max_steps=1)
        model = _model(cfg)
        train_loader, val_loader = _loaders(cfg)
        with pytest.raises(ValueError, match="not causal for next-token"):
            T.train(model, train_loader, cfg, n_steps=1, device=torch.device("cpu"))
        with pytest.raises(ValueError, match="not causal for next-token"):
            T.evaluate(model, val_loader)
    finally:
        _PRIORS.pop(name, None)
        _PRIOR_CAUSALITY.pop(name, None)


def test_baseline_evaluation_does_not_consult_the_vfe_prior(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected(name: str) -> Any:
        raise AssertionError(f"baseline consulted the VFE prior registration ({name!r})")

    monkeypatch.setattr(T, "get_prior_registration", unexpected)
    bcfg = BaselineConfig(vocab_size=V, embed_dim=4, n_heads=2, n_layers=1, mlp_expansion=1,
                          max_seq_len=N, batch_size=B, eval_stride=2, max_steps=1, warmup_steps=0,
                          log_interval=0, eval_interval=0, checkpoint_interval=0, seed=5)
    torch.manual_seed(0)
    model = BaselineTransformer(bcfg)
    _, val_loader = _loaders(_cfg())
    assert math.isfinite(T.evaluate(model, val_loader)["ce"])
