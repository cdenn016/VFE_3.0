r"""Monitoring and reporting remediation of the 2026-10-20 deep audit (lane L2). Tiny CPU models.

Item 12: the crash handler writes ``crash.json`` and the stop line before the emergency checkpoint, and an interrupt
during the checkpoint leaves the record. Item 11 and spike report section 7: the spike attribution records (batch,
position) occurrences, steps since each row's previous occurrence, per-head frame conditions of named phi rows and a
post-update forward state, states coincidences without asserting a cause; the spike-monitor EMA is frozen on spike
steps; the metrics rows carry interval medians; ``block_variance.csv`` carries upper quantiles. Item 14: every-layer
headline and alarm columns. Item 20: allocator and clock telemetry. Item 48: decode and measured FLOPs. Items 7 and
73: test at the validation-selected weights in ``test_results.json``, no empty ``diagnostics`` field.
"""
import csv
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch
from torch.utils.data import DataLoader

import vfe3.run_artifacts as ra
import vfe3.train as T
from tests.profiles import test_config
from vfe3.config import VFE3Config
from vfe3.data.datasets import TokenWindows
from vfe3.model.model import VFEModel

B, N, V = 2, 4, 7
CPU = torch.device("cpu")


@pytest.fixture(autouse=True)
def _no_figure_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(T, "_save_eval_attention_maps", lambda *args, **kwargs: None)
    monkeypatch.setattr(ra, "_run_figures_isolated", lambda *args, **kwargs: True)
    torch.set_num_threads(1)


def _cfg(**overrides: Any) -> VFE3Config:
    values: Dict[str, Any] = dict(
        vocab_size=V, embed_dim=4, n_heads=2, batch_size=B, max_seq_len=N, eval_stride=None,
        n_layers=1, n_e_steps=1, warmup_steps=0, max_steps=8, log_interval=0, eval_interval=0,
        checkpoint_interval=0, skip_belief_sigma_update=True, generate_embedding_figures=False,
        use_block_mlp=False, family="gaussian_diagonal", gauge_group="block_glk",
        attention_mode="projected_predictive", attention_residual=True, decode_mode="linear")
    values.update(overrides)
    return test_config(**values)


def _loaders() -> tuple:
    train = TokenWindows((torch.arange(61) * 5 + 2) % V, N)                 # 15 windows: 7 batches per epoch
    val = TokenWindows((torch.arange(23) * 3) % V, N, pad_final=True)
    return (DataLoader(train, batch_size=B, shuffle=False, drop_last=True),
            DataLoader(val, batch_size=2, shuffle=False))


def _run(cfg: VFE3Config, run_dir: Path, *, n_steps: int, log_interval: int = 0, eval_interval: int = 0,
         model_hook: Any = None, save_model_artifacts: bool = True) -> tuple:
    torch.manual_seed(5)
    model = VFEModel(cfg)
    if model_hook is not None:
        model_hook(model)
    artifacts = ra.RunArtifacts(run_dir, cfg, model, save_model_artifacts=save_model_artifacts)
    train_loader, val_loader = _loaders()
    torch.manual_seed(11)
    losses = T.train(model, train_loader, cfg, n_steps=n_steps, grad_clip=cfg.grad_clip,
                     log_interval=log_interval, eval_interval=eval_interval, val_loader=val_loader, device=CPU,
                     logger=logging.getLogger("audit-l2"), artifacts=artifacts, generate_samples=False)
    return losses, artifacts, model


def _fail_forward_at(fail_on: int) -> Any:
    def hook(model: VFEModel) -> None:
        original = model.forward
        calls: List[int] = []

        def forward(*args: Any, **kwargs: Any) -> Any:
            calls.append(1)
            if len(calls) - 1 == fail_on:
                raise ValueError("injected forward failure")
            return original(*args, **kwargs)
        model.forward = forward
    return hook


def _rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# ------------------------------------------------------------------------------------------------ item 12

def test_crash_record_and_stop_line_precede_the_checkpoint_and_survive_an_interrupt(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "crashed"
    seen_at_save: Dict[str, Any] = {}

    def interrupted_save(self: Any, *args: Any, **kwargs: Any) -> Any:
        record_path = run_dir / "diagnostics" / T.CRASH_RECORD_NAME
        seen_at_save["record"] = json.loads(record_path.read_text(encoding="utf-8"))
        seen_at_save["log"] = (run_dir / T.RUN_LOG_NAME).read_text(encoding="utf-8")
        raise KeyboardInterrupt                                           # a second interrupt during the save

    monkeypatch.setattr(ra.RunArtifacts, "save_checkpoint", interrupted_save)
    with pytest.raises(ValueError, match="injected forward failure"):    # the original exception still wins
        _run(_cfg(), run_dir, n_steps=6, model_hook=_fail_forward_at(3))
    preliminary = seen_at_save["record"]
    assert preliminary["exception"]["type"] == "ValueError" and preliminary["step"] == 4
    assert preliminary["record_complete"] is False and len(preliminary["recent_losses"]) == 3
    assert preliminary["exception"]["traceback"]
    assert "training stopped at step 4" in seen_at_save["log"]
    final = json.loads((run_dir / "diagnostics" / T.CRASH_RECORD_NAME).read_text(encoding="utf-8"))
    assert final["interrupted_stage"] == "checkpoint" and final["record_complete"] is False
    assert final["checkpoint"] is None and any("KeyboardInterrupt" in e for e in final["handler_errors"])
    assert final["failed_batch"] is not None and Path(final["failed_batch"]).is_file()
    assert "interrupted during the checkpoint stage" in (run_dir / T.RUN_LOG_NAME).read_text(encoding="utf-8")


def test_an_uninterrupted_crash_record_is_marked_complete(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        _run(_cfg(), tmp_path / "crashed", n_steps=6, model_hook=_fail_forward_at(3))
    record = json.loads((tmp_path / "crashed" / "diagnostics" / T.CRASH_RECORD_NAME).read_text(encoding="utf-8"))
    assert record["record_complete"] is True and record["interrupted_stage"] is None
    assert record["checkpoint_step"] == 3


# ------------------------------------------------------------------------------------------------ monitor

def _status(norm: float) -> Dict[str, Any]:
    return {"ce": 4.0, "role_grad_norms": {"mu": norm}, "clip_coefs": {}}


def test_norm_ema_is_frozen_on_spike_steps_so_a_burst_stays_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(T, "DIVERGENCE_SPIKE_COOLDOWN", 1)
    monitor = T._DivergenceMonitor()
    for step in range(1, T.DIVERGENCE_EMA_WARMUP + 1):
        monitor.observe(step, 4.0, _status(1.0))
    baseline = monitor._norm_ema["mu"]
    flagged = []
    for step in range(T.DIVERGENCE_EMA_WARMUP + 1, T.DIVERGENCE_EMA_WARMUP + 201):
        monitor.observe(step, 4.0, _status(1e4))                           # a sustained burst at 1e4x
        flagged.append(monitor.event is not None)
    assert all(flagged)                                                      # the old update ratcheted 1.18x a step
    assert monitor._norm_ema["mu"] == baseline
    assert monitor._norm_seen["mu"] == T.DIVERGENCE_EMA_WARMUP + 200        # the warmup count still advances


def test_loss_ema_is_frozen_on_a_loss_spike() -> None:
    monitor = T._DivergenceMonitor()
    for step in range(1, T.DIVERGENCE_EMA_WARMUP + 1):
        monitor.observe(step, 4.0, _status(1.0))
    before = monitor._loss_ema
    monitor.observe(T.DIVERGENCE_EMA_WARMUP + 1, 1e9, _status(1.0))
    assert monitor.spike_loss and monitor._loss_ema == before


def test_interval_median_ignores_a_blowup_and_survives_the_checkpoint_state() -> None:
    monitor = T._DivergenceMonitor()
    for step, loss in enumerate((4.0, 4.2, 1e9, 4.1), start=1):
        monitor.observe(step, loss, {"ce": loss, "role_grad_norms": {}, "clip_coefs": {}})
    columns = monitor.columns({}, flagged=False)
    assert columns["train_loss_interval_median"] == pytest.approx(4.15)
    assert columns["train_ce_interval_median"] == pytest.approx(4.15)
    assert columns["train_loss_interval_mean"] > 1e8
    assert set(T.DIVERGENCE_COLUMNS) == set(columns)
    state = monitor.state_dict(interval_ends=False)
    restored = T._DivergenceMonitor()
    assert restored.load_state_dict(state)
    assert restored.columns({}, flagged=False)["train_loss_interval_median"] == pytest.approx(4.15)
    legacy = json.loads(json.dumps(state))
    del legacy["interval"]["loss_values"], legacy["interval"]["ce_values"]
    older = T._DivergenceMonitor()
    assert older.load_state_dict(legacy)                                     # an older checkpoint still loads
    assert math.isnan(older.columns({}, flagged=False)["train_loss_interval_median"])


def test_metrics_rows_carry_the_interval_medians(tmp_path: Path) -> None:
    losses, _, _ = _run(_cfg(), tmp_path / "run", n_steps=6, log_interval=3, save_model_artifacts=False)
    rows = _rows(tmp_path / "run" / "metrics.csv")
    for row, window in zip(rows, (losses[0:3], losses[3:6])):
        assert float(row["train_loss_interval_median"]) == pytest.approx(sorted(window)[1], rel=1e-12)


# ------------------------------------------------------------------------------------------------ item 11

def test_attribution_reports_positions_recency_and_phi_row_conditions() -> None:
    cfg = _cfg()
    torch.manual_seed(0)
    model = VFEModel(cfg)
    optimizer = T.build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    tokens = torch.tensor([[1, 2, 3, 1], [4, 1, 5, 6]])
    targets = torch.tensor([[2, 3, 1, 4], [1, 5, 6, -100]])
    status: Dict[str, Any] = {}
    T.train_step(model, optimizer, scheduler, tokens, targets, grad_clip=None, status_out=status)
    last_seen = {"token": torch.full((V,), -1, dtype=torch.int64),
                 "target": torch.full((V + 1,), -1, dtype=torch.int64)}
    last_seen["token"][1] = 7
    report = T._spike_attribution(model, optimizer, status, tokens, roles=("mu", "sigma", "phi"),
                                  loss_spiked=True, replayed=False, last_group_norms={}, targets=targets,
                                  step=10, last_seen=last_seen)
    assert all("hard batch" not in cause for cause in report["causes"])
    assert "the training loss spiked on the same step" in report["causes"]
    tables = [entry for entry in report["table_rows"].values() if entry["indexed_by"] == "token"]
    assert tables
    for entry in tables:
        for row, found, since in zip(entry["rows"], entry["occurrences"], entry["steps_since_last_occurrence"]):
            assert found == [[b, n] for b, n in (tokens == row).nonzero().tolist()][:T._SPIKE_ROW_OCCURRENCES]
            assert since == (3 if row == 1 else None)
    phi_report = T._spike_attribution(model, optimizer, status, tokens, roles=("phi",), loss_spiked=False,
                                      replayed=False, last_group_norms={}, targets=targets)
    phi = [entry for name, entry in phi_report["table_rows"].items() if name.endswith(".phi_embed")]
    assert phi, "the phi role's top parameters include the token frame table"
    if phi:
        (entry,) = phi
        conditions = entry["head_condition"]
        assert len(conditions) == len(entry["rows"]) and all(len(c) == cfg.n_heads for c in conditions)
        assert all(value >= 1.0 - 1e-9 for c in conditions for value in c)
        # The identity frame has condition 1 per head.
        assert T._phi_row_head_conditions(model, torch.zeros(1, model.prior_bank.phi_embed.shape[1])) == [
            pytest.approx([1.0] * cfg.n_heads)]


def test_forward_state_reports_the_largest_variance_and_per_position_ce() -> None:
    cfg = _cfg()
    torch.manual_seed(0)
    model = VFEModel(cfg)
    tokens = torch.tensor([[1, 2, 3, 1], [4, 1, 5, 6]])
    targets = torch.tensor([[2, 3, 1, 4], [1, 5, 6, -100]])
    rng = torch.get_rng_state()
    state = T._spike_forward_state(model, tokens, targets)
    assert torch.equal(torch.get_rng_state(), rng) and model.training
    model.eval()
    with torch.no_grad():
        belief, logits = model.forward_beliefs(tokens, return_logits=True)
    variance = belief.sigma.amax(-1)                                         # (B, N)
    b, n = state["sigma_q_max_at"]
    assert state["sigma_q_max"] == pytest.approx(float(variance.max()), rel=1e-5)
    assert float(variance[b, n]) == pytest.approx(float(variance.max()), rel=1e-5)
    ce = torch.nn.functional.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), ignore_index=-100,
                                           reduction="none").reshape(B, N)
    ce[targets == -100] = float("-inf")
    assert state["ce_max"] == pytest.approx(float(ce.max()), rel=1e-5)
    assert state["ce_max_at"] == [int(i) for i in divmod(int(ce.reshape(-1).argmax()), N)]


def test_a_reported_spike_record_carries_the_forward_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                           caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setattr(T, "DIVERGENCE_EMA_WARMUP", 1)
    monkeypatch.setattr(T, "DIVERGENCE_NORM_SPIKE_RATIO", 1e-9)              # every step's role norm spikes
    with caplog.at_level(logging.WARNING, logger="audit-l2"):
        _run(_cfg(), tmp_path / "run", n_steps=4, log_interval=4, save_model_artifacts=False)
    records = [json.loads(line) for line in (tmp_path / "run" / "diagnostics" / T.DIVERGENCE_SPIKE_LOG)
               .read_text(encoding="utf-8").splitlines()]
    reported = [record for record in records if record["reported"]]
    assert reported and all("forward_state" in record for record in reported)
    assert all("forward_state" not in record for record in records if not record["reported"])
    state = reported[0]["forward_state"]
    assert math.isfinite(state["sigma_q_max"]) and len(state["sigma_q_max_at"]) == 2
    assert "loss_over_3_log_v" in reported[0]
    lines = [r.getMessage() for r in caplog.records if "spike attribution" in r.getMessage()]
    assert lines and "max sigma_q" in lines[0] and "hard batch" not in lines[0]


# ------------------------------------------------------------------------------------------------ block variance

def test_block_variance_rows_carry_upper_quantiles() -> None:
    variances = torch.cat([torch.ones(998), torch.tensor([1e3, 1e6])])
    slot = {(0, "pre_norm"): [[variances], [torch.ones(4)]]}
    (row,) = T._block_variance_rows(slot, 3)
    assert set(row) == set(T.BLOCK_VARIANCE_FIELDS)
    assert row["var_q99"] == pytest.approx(float(torch.quantile(variances.double(), 0.99)))
    assert row["var_q999"] == pytest.approx(float(torch.quantile(variances.double(), 0.999)))
    assert row["var_median"] <= row["var_q99"] <= row["var_q999"] <= row["var_max"]


def test_an_older_block_variance_file_is_migrated_on_append(tmp_path: Path) -> None:
    cfg = _cfg()
    artifacts = ra.RunArtifacts(tmp_path / "run", cfg, VFEModel(cfg), save_model_artifacts=False)
    old_fields = [field for field in T.BLOCK_VARIANCE_FIELDS if field not in ("var_q99", "var_q999")]
    path = artifacts.run_dir / "diagnostics" / "block_variance.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=old_fields)
        writer.writeheader()
        writer.writerow({field: 1 for field in old_fields})
    new_row = {field: 2 for field in T.BLOCK_VARIANCE_FIELDS}
    artifacts.append_diagnostic_rows("block_variance.csv", T.BLOCK_VARIANCE_FIELDS, [new_row])
    rows = _rows(path)
    assert list(rows[0]) == list(T.BLOCK_VARIANCE_FIELDS)
    assert rows[0]["var_q99"] == "" and rows[0]["var_min"] == "1" and rows[1]["var_q999"] == "2"
    with pytest.raises(ValueError):                                          # a different schema still raises
        artifacts.append_diagnostic_rows("block_variance.csv", ["step", "other"], [{"step": 1, "other": 2}])


# ------------------------------------------------------------------------------------------------ item 14

def test_every_layer_columns_and_the_per_layer_eigenvalue_alarm(caplog: pytest.LogCaptureFixture) -> None:
    cfg = _cfg(n_layers=2)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    tokens = torch.randint(0, V, (B, N), generator=torch.Generator().manual_seed(3))
    d = model.diagnostics(tokens)
    extremes = model.last_belief_layer_extremes
    assert len(extremes["lam_min"]) == len(extremes["lam_max"]) == 2
    assert d["belief_lam_min_layer_min"] == pytest.approx(min(extremes["lam_min"]))
    assert d["belief_lam_max_layer_max"] == pytest.approx(max(extremes["lam_max"]))
    assert d["belief_lam_min_layer_min"] <= d["belief_lam_min"]               # the last layer is one of the layers
    ratios = [low / high for low, high in zip(extremes["lam_min"], extremes["lam_max"])]
    assert d["belief_eig_ratio_layer_min"] == pytest.approx(min(ratios))
    assert d["belief_eig_ratio_argmin_layer"] == float(ratios.index(min(ratios)))
    for key in ("relational_capped_mass_mean_layer_max", "relational_capped_mass_mean_argmax_layer",
                "relational_score_term_query_constant_median_layer_max", "belief_cond_max_layer_max"):
        assert key in d
    assert d["relational_capped_mass_mean_layer_max"] >= d["relational_capped_mass_mean"] or math.isnan(
        d["relational_capped_mass_mean"])
    # The alarm reads every layer: layer 0 collapsed, the last layer healthy.
    warned: set = set()
    records: List[Dict[str, Any]] = []
    with caplog.at_level(logging.WARNING):
        T._warn_eigenvalue_collapse({"belief_lam_min": 1.0, "belief_lam_max": 2.0}, warned,
                                    logging.getLogger("audit-l2"), belief_layer=1,
                                    layer_extremes={"lam_min": [1e-9, 1.0], "lam_max": [10.0, 2.0]},
                                    sink=records.append)
    assert ("belief_eigenvalue_ratio", 0, -1) in warned and ("belief_eigenvalue_ratio", 1, -1) not in warned


def test_floor_reference_is_armed_under_the_max_floor() -> None:
    cfg = _cfg(predictive_covariance="moment_matched", predictive_covariance_floor_mode="max")
    torch.manual_seed(0)
    model = VFEModel(cfg)
    prior = VFEModel(_cfg(predictive_covariance="moment_matched", predictive_covariance_floor_mode="prior"))
    if T._covariance_floor_reference(prior) is None:
        pytest.skip("this tiny route has no diagonal prior table")
    from vfe3.numerics import bounded_variance_from_log
    bank = model.prior_bank
    expected = cfg.predictive_covariance_floor * float(
        bounded_variance_from_log(bank._prior_sigma_log_table().double(), eps=bank.eps).min())
    assert T._covariance_floor_reference(model) == pytest.approx(expected)


# ------------------------------------------------------------------------------------------------ item 20

def test_device_telemetry_is_nan_off_cuda_and_summarized_tolerantly(tmp_path: Path) -> None:
    telemetry = T._DeviceTelemetry(CPU).sample()
    assert set(telemetry) == set(T.DEVICE_TELEMETRY_COLUMNS) and all(math.isnan(v) for v in telemetry.values())
    from vfe3.metrics_schema import METRIC_COLUMNS
    from vfe3.run_resources import _TELEMETRY_COLUMNS, throughput_summary
    assert set(_TELEMETRY_COLUMNS) == set(T.DEVICE_TELEMETRY_COLUMNS) <= set(METRIC_COLUMNS)
    old = tmp_path / "old"
    old.mkdir()
    (old / "metrics.csv").write_text("step,tokens_per_s,peak_mem_mb\n10,100.0,5.0\n", encoding="utf-8")
    summary = throughput_summary(old)
    assert summary["device_telemetry"]["gpu_sm_clock_mhz"]["first"] is None       # an older run: no columns
    new = tmp_path / "new"
    new.mkdir()
    (new / "metrics.csv").write_text("step,tokens_per_s,gpu_sm_clock_mhz,cuda_alloc_retries\n"
                                     "10,100.0,2700,0\n20,99.0,2650,3\n", encoding="utf-8")
    summary = throughput_summary(new)
    assert summary["device_telemetry"]["gpu_sm_clock_mhz"]["first"] == 2700.0
    assert summary["device_telemetry"]["cuda_alloc_retries"]["last"] == 3.0


def test_metrics_rows_carry_the_telemetry_columns(tmp_path: Path) -> None:
    _run(_cfg(), tmp_path / "run", n_steps=2, log_interval=2, save_model_artifacts=False)
    (row,) = _rows(tmp_path / "run" / "metrics.csv")
    assert all(column in row for column in T.DEVICE_TELEMETRY_COLUMNS)


# ------------------------------------------------------------------------------------------------ item 48

def test_measured_forward_flops_split_the_decode() -> None:
    cfg = _cfg(decode_mode="diagonal_chunked")
    torch.manual_seed(0)
    model = VFEModel(cfg)
    fields = ra._measured_forward_flops(model, cfg, tokens_seen=100)
    assert fields["flops_per_token_forward_measured"] > fields["flops_per_token_beliefs_measured"] > 0
    assert fields["flops_per_token_decode_measured"] == pytest.approx(
        fields["flops_per_token_forward_measured"] - fields["flops_per_token_beliefs_measured"])
    # The dense prior-bank decode is the (N, 2K) x (2K, V) product: 4VK per token.
    assert fields["flops_per_token_decode_measured"] >= 4 * V * cfg.embed_dim
    assert fields["est_train_flops_measured_3x"] == pytest.approx(
        3 * fields["flops_per_token_forward_measured"] * 100)
    assert model.training


# ------------------------------------------------------------------------------------------------ items 7, 73

def test_test_results_carry_the_validation_selected_test_and_no_empty_diagnostics(
        tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    cfg = _cfg()
    losses, artifacts, model = _run(cfg, tmp_path / "run", n_steps=4, eval_interval=2)
    train_loader, val_loader = _loaders()
    test_loader = DataLoader(TokenWindows((torch.arange(19) * 5 + 1) % V, N, pad_final=True),
                             batch_size=2, shuffle=False)
    with caplog.at_level(logging.INFO, logger="audit-l2"):
        results = ra.finalize_run(model, artifacts, cfg, final_validation=T.evaluate(model, val_loader),
                                  train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                                  losses=losses, device=CPU, logger=logging.getLogger("audit-l2"))
    test_results = json.loads((artifacts.run_dir / "test_results.json").read_text(encoding="utf-8"))
    summary = json.loads((artifacts.run_dir / "summary.json").read_text(encoding="utf-8"))
    assert "diagnostics" not in test_results and "diagnostics" not in summary
    assert test_results["test_best_ppl"] is not None and test_results["test_best_step"] == artifacts.best_step
    assert test_results["test_best_ppl"] == summary["test_best_ppl"] and results["test_ppl"] == summary["test_ppl"]
    assert any("at the validation-selected weights" in r.getMessage() for r in caplog.records)
