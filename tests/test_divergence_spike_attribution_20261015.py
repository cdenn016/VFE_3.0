r"""Attribution of a divergence spike to parameters, groups and table rows (2026-10-15).

A spike line named only the role (mu, sigma or phi), and the forced metrics row left every per-group gradient column
blank, so a phi spike could not be traced to the token frame table, the positional frames or a layer. On a spike
step the training loop now reads the live gradients (train_step zeroes on entry, so they survive the step), undoes
each parameter's clip coefficient, and reports the parameters, named groups and token-table rows that carry the
spiking role's pre-clip norm, with the step's coincident causes. Tiny CPU model.
"""
import csv
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch

import vfe3.run_artifacts as ra
import vfe3.train as T
from tests.test_audit2_20261011_training import _batch, _cfg, _loaders, _model, _train


@pytest.fixture(autouse=True)
def _no_figure_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(T, "_save_eval_attention_maps", lambda *args, **kwargs: None)
    monkeypatch.setattr(ra, "_run_figures_isolated", lambda *args, **kwargs: True)
    torch.set_num_threads(1)


def _rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.parametrize("grad_clip", [None, 1e-3])
def test_attribution_recovers_the_pre_clip_role_norm(grad_clip: Any) -> None:
    cfg = _cfg(grad_clip=grad_clip)
    model = _model(cfg)
    optimizer = T.build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    status: Dict[str, Any] = {}
    tokens, targets = _batch(cfg)
    T.train_step(model, optimizer, scheduler, tokens, targets, grad_clip=grad_clip, status_out=status)
    report = T._spike_attribution(model, optimizer, status, tokens, roles=("mu", "sigma", "phi"),
                                  loss_spiked=False, replayed=False, last_group_norms={})
    for role, norm in status["role_grad_norms"].items():
        if role not in report["roles"]:
            continue
        assert report["roles"][role]["pre_clip_norm"] == pytest.approx(norm, rel=1e-5)
        shares = report["roles"][role]["group_shares"]
        assert sum(shares.values()) == pytest.approx(1.0, rel=1e-6)
    top = report["top_parameters"]
    assert top and all({"name", "role", "group", "pre_clip_norm", "role_share"} <= set(entry) for entry in top)
    assert [entry["pre_clip_norm"] for entry in top] == sorted((e["pre_clip_norm"] for e in top), reverse=True)


def test_token_table_rows_are_reported_with_their_batch_counts() -> None:
    cfg = _cfg()
    model = _model(cfg)
    optimizer = T.build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    status: Dict[str, Any] = {}
    tokens, targets = _batch(cfg)
    T.train_step(model, optimizer, scheduler, tokens, targets, grad_clip=None, status_out=status)
    report = T._spike_attribution(model, optimizer, status, tokens, roles=("mu", "sigma", "phi"),
                                  loss_spiked=False, replayed=False, last_group_norms={})
    rows = report["table_rows"]
    assert rows, "a token table with gradients must be reported"
    counts = torch.bincount(tokens.reshape(-1), minlength=cfg.vocab_size)
    for entry in rows.values():
        assert 0.0 < entry["top_share"] <= 1.0 + 1e-9
        if entry["indexed_by"] == "token":
            for row, count in zip(entry["rows"], entry["batch_counts"]):
                assert count == int(counts[row])


def test_a_spike_logs_attribution_writes_the_record_and_fills_the_group_columns(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setattr(T, "DIVERGENCE_EMA_WARMUP", 2)
    real_step = T.train_step
    calls = {"n": 0}

    def spiking(*args: Any, **kwargs: Any) -> float:
        loss = real_step(*args, **kwargs)
        calls["n"] += 1
        if calls["n"] == 5:
            kwargs["status_out"]["role_grad_norms"]["phi"] *= 1e3
        return loss

    monkeypatch.setattr(T, "train_step", spiking)
    cfg = _cfg(max_steps=8)
    model = _model(cfg)
    artifacts = ra.RunArtifacts(tmp_path / "run", cfg, model, save_model_artifacts=False)
    with caplog.at_level(logging.WARNING, logger="audit2-1011"):
        _train(model, cfg, _loaders(cfg), artifacts=artifacts, log_interval=4, n_steps=8)
    lines = [record.getMessage() for record in caplog.records if "spike attribution" in record.getMessage()]
    assert len(lines) == 1 and "step 5" in lines[0] and "phi" in lines[0]
    records = [json.loads(line) for line in
               (tmp_path / "run" / "diagnostics" / T.DIVERGENCE_SPIKE_LOG).read_text(encoding="utf-8").splitlines()]
    assert [r["step"] for r in records] == [5]
    assert records[0]["spiking_roles"] == ["phi"] and records[0]["top_parameters"]
    forced = _rows(tmp_path / "run" / "metrics.csv")[1]
    assert int(forced["step"]) == 5
    assert any(math.isfinite(float(forced[c])) for c in T._GRAD_GROUP_COLUMNS if forced.get(c, "") != "")


def test_decode_table_rows_are_counted_over_the_targets() -> None:
    r"""Untied decode tables are read at the targets, so their rows are reported with target counts
    (audit 2026-10-15, finding 8: decode_sigma_log_embed topped every sigma spike with no rows listed)."""
    cfg = _cfg(decode_mode="diagonal_chunked", untie_decode_bank=True)
    model = _model(cfg)
    optimizer = T.build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    status: Dict[str, Any] = {}
    tokens, targets = _batch(cfg)
    T.train_step(model, optimizer, scheduler, tokens, targets, grad_clip=None, status_out=status)
    report = T._spike_attribution(model, optimizer, status, tokens, roles=("mu", "sigma", "phi"),
                                  loss_spiked=False, replayed=False, last_group_norms={}, targets=targets)
    decode = {name: entry for name, entry in report["table_rows"].items()
              if name.rsplit(".", 1)[-1] in T._SPIKE_DECODE_TABLES}
    top_names = {entry["name"] for entry in report["top_parameters"]}
    expected = {name for name in top_names if name.rsplit(".", 1)[-1] in T._SPIKE_DECODE_TABLES}
    assert expected, "the tiny case must put a decode table among the top parameters"
    assert set(decode) == expected
    counts = torch.bincount(targets.reshape(-1), minlength=cfg.vocab_size)
    for entry in decode.values():
        assert entry["indexed_by"] == "target"
        assert entry["batch_counts"] == [int(counts[row]) for row in entry["rows"]]
