r"""Training-loop monitors of the 2026-10-16 deep audit, lane C1 (findings 4, 12, 13, 51 and 110).

F4: an optimizer that clears ``.grad`` inside ``step()`` (the table Fisher step on ``mu_embed`` and
``sigma_log_embed``; the manifold step on ``phi_embed`` and ``pos_phi_free`` under ``pullback_group``)
hid those tables from the spike attribution and from the known-dead set of the parameter-motion check,
both of which read ``p.grad`` after the step. ``train_step`` now keeps references to the gradients that
reached ``optimizer.step`` in ``status_out["grads"]`` and both readers consult them.

F12: the decode variance table had no training-time gauge (``decode_sigma_v_min``, which the
``decode_av_precision`` comment says to watch, had no caller), and the logged ``sigma_log_*`` columns read the
encoder table only. Every metrics row now carries ``decode_sigma_v_min`` and the decode table's log-variance
mean and standard deviation, NaN under the linear readout.

F13: under ``m_table_update_mode='fisher'`` the ``update_rms_per_lr_group_*`` columns divided the table
step by the AdamW rate the config marks inert; they now divide by the realized Fisher step size
:math:`\eta\,\mathrm{lr}_t/\mathrm{lr}_0`.

F51: with an enabled GradScaler and clipping off, ``train_step`` published no role norms, so the norm-spike
monitor was silent; asking for ``status_out`` under an enabled scaler now unscales once and publishes the
set norms in true units.

F110: ``parameter_report`` counted a parameter whose probe gradient is identically zero as live without
saying so; it now lists such parameters under ``zero_grad_names`` beside ``dead_names``.

Tiny CPU models only (K=4, two heads, N<=6, vocab <= 9).
"""
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


def _route_cfg(table_mode: str, phi_mode: str, **overrides: Any) -> Any:
    r"""The tiny training config on a chosen table and phi update route (fisher needs an untied decode bank)."""
    return _cfg(decode_mode="diagonal_chunked", untie_decode_bank=True,
                m_table_update_mode=table_mode, m_phi_update_mode=phi_mode, **overrides)


_ROUTES: Tuple[Tuple[str, str], ...] = (
    ("adamw", "adamw"), ("fisher", "adamw"), ("adamw", "pullback_group"), ("fisher", "pullback_group"))


def _one_step(cfg: Any, grad_clip: Any, **step_kwargs: Any) -> Tuple[Any, Any, Dict[str, Any], torch.Tensor,
                                                                    torch.Tensor]:
    model = _model(cfg)
    optimizer = T.build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    status: Dict[str, Any] = {}
    tokens, targets = _batch(cfg)
    T.train_step(model, optimizer, scheduler, tokens, targets, grad_clip=grad_clip, status_out=status,
                 **step_kwargs)
    return model, optimizer, status, tokens, targets


# ------------------------------------------------------------------------------------------------- F4

@pytest.mark.parametrize("grad_clip", [None, 1e-3])
@pytest.mark.parametrize("table_mode,phi_mode", _ROUTES)
def test_attribution_sees_the_tables_an_optimizer_clears(table_mode: str, phi_mode: str, grad_clip: Any) -> None:
    cfg = _route_cfg(table_mode, phi_mode)
    model, optimizer, status, tokens, targets = _one_step(cfg, grad_clip)
    assert status["did_step"]
    report = T._spike_attribution(model, optimizer, status, tokens, roles=("mu", "sigma", "phi"),
                                  loss_spiked=False, replayed=False, last_group_norms={}, targets=targets)
    for role, norm in status["role_grad_norms"].items():
        assert role in report["roles"], f"role {role} detected at {norm} but absent from the attribution"
        assert report["roles"][role]["pre_clip_norm"] == pytest.approx(norm, rel=1e-5), role
        assert sum(report["roles"][role]["group_shares"].values()) == pytest.approx(1.0, rel=1e-6)
    expected = {"mu": {"mu_embed"}, "sigma": {"sigma_log_embed"}, "phi": {"phi_embed", "pos_phi"}}
    for role, groups in expected.items():
        assert groups <= set(report["roles"][role]["group_shares"]), (role, report["roles"][role])
        assert groups <= set(report["group_norms"]), role


@pytest.mark.parametrize("table_mode,phi_mode", _ROUTES)
def test_known_dead_set_keeps_the_tables_an_optimizer_clears(table_mode: str, phi_mode: str) -> None:
    cfg = _route_cfg(table_mode, phi_mode)
    model, _, status, _, _ = _one_step(cfg, 1.0)
    dead = T._known_dead_parameters(model, status)
    for name in ("prior_bank.mu_embed", "prior_bank.sigma_log_embed", "prior_bank.phi_embed", "pos_phi_free"):
        assert name not in dead, name
    # Without the stash the classification falls back to the live gradients.
    model.zero_grad(set_to_none=True)
    fallback = T._known_dead_parameters(model, {})
    assert set(fallback) == {name for name, p in model.named_parameters() if p.requires_grad}


def test_fisher_spike_row_and_motion_report_carry_the_tables(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(T, "DIVERGENCE_EMA_WARMUP", 2)
    real_step = T.train_step
    calls = {"n": 0}

    def spiking(*args: Any, **kwargs: Any) -> float:
        loss = real_step(*args, **kwargs)
        calls["n"] += 1
        if calls["n"] == 5:
            kwargs["status_out"]["role_grad_norms"]["mu"] *= 1e3
        return loss

    monkeypatch.setattr(T, "train_step", spiking)
    cfg = _route_cfg("fisher", "adamw", max_steps=8)
    model = _model(cfg)
    artifacts = ra.RunArtifacts(tmp_path / "run", cfg, model, save_model_artifacts=False)
    _train(model, cfg, _loaders(cfg), artifacts=artifacts, log_interval=4, n_steps=8)
    records = [json.loads(line) for line in
               (tmp_path / "run" / "diagnostics" / T.DIVERGENCE_SPIKE_LOG).read_text(encoding="utf-8").splitlines()]
    assert [r["step"] for r in records] == [5]
    assert {"mu_embed", "sigma_log_embed"} <= set(records[0]["group_norms"])
    forced = _rows(tmp_path / "run" / "metrics.csv")[1]
    assert int(forced["step"]) == 5
    for column in ("grad_norm_group_mu_embed", "grad_norm_group_sigma_log_embed"):
        assert forced[column] != "" and math.isfinite(float(forced[column])), column
    motion = json.loads((tmp_path / "run" / "parameter_motion.json").read_text(encoding="utf-8"))
    for name in ("prior_bank.mu_embed", "prior_bank.sigma_log_embed"):
        assert motion["motion"][name]["known_dead"] is False, name


# ------------------------------------------------------------------------------------------------ F13

def _fisher_step_columns(m_p_mu_lr: float, m_p_sigma_lr: float) -> Tuple[Dict[str, float], Dict[str, float],
                                                                         Dict[str, float], Dict[str, float]]:
    r"""The step-in-rate-units columns of one Fisher table step at a schedule factor of 0.5."""
    # Zero table decay, as on the live route: the AdamW rate then only schedules the Fisher step (it is inert).
    cfg = _route_cfg("fisher", "adamw", m_p_mu_lr=m_p_mu_lr, m_p_sigma_lr=m_p_sigma_lr,
                     mu_weight_decay=0.0, sigma_weight_decay=0.0)
    model = _model(cfg)
    optimizer = T.build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.5)
    tokens, targets = _batch(cfg)
    before = T._parameter_snapshot(model)
    rates = T._parameter_learning_rates(model, optimizer)
    fisher_rates = T._table_fisher_learning_rates(optimizer)
    T.train_step(model, optimizer, scheduler, tokens, targets, grad_clip=1.0, status_out={})
    out = T._group_update_statistics(model, before, learning_rates=rates)
    deltas = {name: float((dict(model.named_parameters())[name].detach().double() - before[name].double()).norm())
              for name in ("prior_bank.mu_embed", "prior_bank.sigma_log_embed")}
    return out, rates, fisher_rates, deltas


def test_fisher_table_step_columns_divide_by_the_realized_step_size() -> None:
    low, rates, fisher_rates, deltas = _fisher_step_columns(0.025, 0.001)
    high, _, _, high_deltas = _fisher_step_columns(0.25, 0.01)
    eta = 0.1                                                                   # tests/profiles m_table_fisher_lr
    assert fisher_rates == pytest.approx({"mu": 0.5 * eta, "sigma": 0.5 * eta})
    assert rates["prior_bank.mu_embed"] == pytest.approx(fisher_rates["mu"])
    assert rates["prior_bank.sigma_log_embed"] == pytest.approx(fisher_rates["sigma"])
    for group, name in (("mu_embed", "prior_bank.mu_embed"), ("sigma_log_embed", "prior_bank.sigma_log_embed")):
        column = f"update_rms_per_lr_group_{group}"
        assert deltas[name] > 0.0 and deltas[name] == pytest.approx(high_deltas[name], rel=1e-9)   # inert rate
        assert low[column] == pytest.approx(high[column], rel=1e-9), column
        n = dict(_model(_route_cfg("fisher", "adamw")).named_parameters())[name].numel()
        assert low[column] == pytest.approx(deltas[name] / (0.5 * eta * math.sqrt(n)), rel=1e-9), column


# ------------------------------------------------------------------------------------------------ F51

def _scaled_step(grad_clip: Any, *, scaler_on: bool, with_status: bool = True) -> Tuple[Any, Any, Dict[str, Any],
                                                                                         torch.Tensor]:
    cfg = _cfg()
    model = _model(cfg)
    optimizer = T.build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    scaler = torch.amp.GradScaler(device="cpu", init_scale=1024.0, enabled=scaler_on)
    status: Dict[str, Any] = {}
    tokens, targets = _batch(cfg)
    T.train_step(model, optimizer, scheduler, tokens, targets, grad_clip=grad_clip, scaler=scaler,
                 status_out=status if with_status else None)
    return model, optimizer, status, tokens


@pytest.mark.parametrize("grad_clip", [None, 1.0])
def test_enabled_scaler_publishes_true_unit_role_norms(grad_clip: Any) -> None:
    _, _, reference, _ = _scaled_step(grad_clip, scaler_on=False)
    model, optimizer, status, tokens = _scaled_step(grad_clip, scaler_on=True)
    assert status["did_step"] and reference["role_grad_norms"]
    assert status["role_grad_norms"].keys() == reference["role_grad_norms"].keys()
    for role, norm in reference["role_grad_norms"].items():
        assert status["role_grad_norms"][role] == pytest.approx(norm, rel=1e-5), role
    report = T._spike_attribution(model, optimizer, status, tokens, roles=("mu", "sigma", "phi"),
                                  loss_spiked=False, replayed=False, last_group_norms={})
    for role, norm in reference["role_grad_norms"].items():
        if norm > 0.0:
            assert report["roles"][role]["pre_clip_norm"] == pytest.approx(norm, rel=1e-5), role


def test_enabled_scaler_status_request_leaves_the_update_unchanged() -> None:
    with_status, _, _, _ = _scaled_step(None, scaler_on=True)
    silent, _, _, _ = _scaled_step(None, scaler_on=True, with_status=False)
    for (name, p), (_, q) in zip(with_status.named_parameters(), silent.named_parameters()):
        assert torch.equal(p, q), name


# ----------------------------------------------------------------------------------------------- F110

def test_parameter_report_lists_zero_gradient_parameters_beside_the_dead_ones() -> None:
    from tests.profiles import test_config
    from vfe3.model.model import VFEModel
    cfg = test_config(block_mlp_mode="gauge_gate", use_block_mlp=True, family="gaussian_full")
    torch.manual_seed(0)
    model = VFEModel(cfg)
    rep = T.parameter_report(model)
    assert rep["probed"] and rep["dead_names"] == []
    # The gate's second layer starts at zero, so its first layer is reached by autograd with a zero gradient.
    assert rep["zero_grad_names"] == ["block_mlps.0.fc1.weight", "block_mlps.0.fc1.bias"]
    assert rep["live"] == rep["trainable"]                       # still counted live: the contract is grad-not-None
    assert all(p.grad is None for p in model.parameters())       # the probe leaves no gradients behind
    banner = T._banner(model, cfg, "synthetic", torch.device("cpu"), n_steps=1)
    assert "zero gradient at the init probe" in banner and "block_mlps.0.fc1.weight" in banner
    plain = VFEModel(test_config())
    assert T.parameter_report(plain)["zero_grad_names"] == []
    assert "zero gradient at the init probe" not in T._banner(plain, plain.cfg, "synthetic", torch.device("cpu"), 1)


# ------------------------------------------------------------------------------------------------ F12

def _set_rows(table: torch.Tensor, rows: slice, value: float) -> None:
    with torch.no_grad():
        table[rows] = value


def test_decode_variance_columns_read_the_table_the_decoder_scores_against() -> None:
    from tests.profiles import live_config
    from vfe3.model.model import VFEModel
    # The launcher's untied prior-bank decode (its committed readout may be linear; the decode route is pinned here).
    cfg = live_config(decode_mode="diagonal_chunked", untie_decode_bank=True)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    bank = model.prior_bank
    before = T._sigma_log_statistics(model)
    stats = T._decode_variance_statistics(model)
    assert set(stats) == set(T.DECODE_VARIANCE_COLUMNS)
    assert stats["decode_sigma_v_min"] == pytest.approx(bank.decode_sigma_v_min(), rel=1e-12)
    # Shrinking half of the untied decode table moves the decode columns and leaves the encoder ones alone.
    _set_rows(bank.decode_sigma_log_embed, slice(0, cfg.vocab_size // 2), math.log(1e-5))
    after = T._decode_variance_statistics(model)
    assert after["decode_sigma_v_min"] == pytest.approx(1e-5, rel=1e-5)
    values = bank.decode_sigma_log_embed.detach().double()
    assert after["decode_sigma_log_mean"] == pytest.approx(float(values.mean()), rel=1e-6)
    assert after["decode_sigma_log_sd"] == pytest.approx(float(values.std(correction=0)), rel=1e-6)
    assert T._sigma_log_statistics(model) == before


def test_decode_variance_columns_on_a_tied_bank_and_under_the_linear_readout() -> None:
    from tests.profiles import test_config
    from vfe3.model.model import VFEModel
    torch.manual_seed(0)
    tied = VFEModel(test_config(decode_mode="diagonal_chunked", untie_decode_bank=False))
    _set_rows(tied.prior_bank.sigma_log_embed, slice(0, 2), math.log(1e-3))
    stats = T._decode_variance_statistics(tied)
    assert stats["decode_sigma_v_min"] == pytest.approx(1e-3, rel=1e-5)
    encoder = T._sigma_log_statistics(tied)
    assert stats["decode_sigma_log_mean"] == pytest.approx(encoder["sigma_log_mean"], rel=1e-12)
    assert stats["decode_sigma_log_sd"] == pytest.approx(encoder["sigma_log_sd"], rel=1e-12)
    linear = VFEModel(test_config(decode_mode="linear"))
    assert all(math.isnan(value) for value in T._decode_variance_statistics(linear).values())


@pytest.mark.parametrize("decode_mode", ["diagonal_chunked", "linear"])
def test_decode_variance_columns_reach_metrics_csv(tmp_path: Path, decode_mode: str) -> None:
    cfg = _cfg(decode_mode=decode_mode, untie_decode_bank=decode_mode != "linear", max_steps=4)
    model = _model(cfg)
    artifacts = ra.RunArtifacts(tmp_path / "run", cfg, model, save_model_artifacts=False)
    _train(model, cfg, _loaders(cfg), artifacts=artifacts, log_interval=2, n_steps=4)
    rows = _rows(tmp_path / "run" / "metrics.csv")
    assert [int(row["step"]) for row in rows] == [2, 4]
    for row in rows:
        for column in T.DECODE_VARIANCE_COLUMNS:
            if decode_mode == "linear":
                assert row[column] == "", column
            else:
                assert math.isfinite(float(row[column])), column
        if decode_mode != "linear":
            assert float(row["decode_sigma_v_min"]) > 0.0
    assert artifacts.diagnostics_failures == {}
