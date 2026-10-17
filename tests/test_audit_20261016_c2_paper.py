r"""Audit 2026-10-16, lane C2 (paper tables and figures): F21, F22, F67.

Tiny synthetic runs only (vocabulary 6, K = 4, two heads, one layer, CPU, one thread); nothing is
trained. A real ``TableFisherAdamW`` is built by :func:`vfe3.train.build_optimizer` under
``m_table_update_mode='fisher'`` so the checkpoint carries the executed ``table_fisher`` group markers.

* F21: under the Fisher table M-step ``metrics.csv`` ``lr_sigma`` logs the natural step
  :math:`\eta\,\mathrm{lr}_t/\mathrm{lr}_0`, while :mod:`vfe3.table_optim` decays the table at the AdamW
  group rate :math:`\mathrm{lr}_t`; the S1 decay-only reference must use :math:`\mathrm{lr}_t`.
* F22: the methods-contract Optimization field reports the Fisher table step and the realized
  ``grad_clip_scope``, with the ``table_fisher`` groups exempt from the clip.
* F67: M2 and the decision-rule table mark a shared data order and config differences visibly.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pytest
import torch

from tests.test_paper_20260928 import _cfg, _write_run
from vfe3.paper import figures as pf
from vfe3.paper import runs as pr
from vfe3.paper import tables as pt
from vfe3.paper.runs import Arm, Manifest, RunRecord

ETA, RADIUS, SIGMA_LR, SIGMA_WD = 0.1, 0.5, 0.001, 0.01


@pytest.fixture(autouse=True)
def _one_thread() -> None:
    torch.set_num_threads(1)


def _build(cfg: Any) -> Dict[str, Any]:
    from vfe3.model.model import VFEModel
    from vfe3.run_artifacts import _optimizer_parameter_manifest
    from vfe3.train import _floor_lr_lambdas, build_optimizer

    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(cfg).eval()
    optimizer = build_optimizer(model, cfg)
    base = [group["lr"] for group in optimizer.param_groups]
    torch.optim.lr_scheduler.LambdaLR(optimizer, _floor_lr_lambdas(base, cfg))
    state = optimizer.state_dict()
    manifest = _optimizer_parameter_manifest(model, optimizer, state)
    model_state = {name: t.detach().clone() for name, t in model.state_dict().items()}
    return {"cfg": cfg, "state": model_state, "optimizer_state": state, "manifest": manifest,
            "optimizer": optimizer}


@pytest.fixture(scope="module")
def fisher() -> Dict[str, Any]:
    r"""A tiny fisher-mode model with per-group clipping, its TableFisherAdamW groups and scheduler."""
    cfg = _cfg(m_table_update_mode="fisher", decode_mode="linear", m_table_fisher_lr=ETA,
               m_table_fisher_trust_radius=RADIUS, m_p_sigma_lr=SIGMA_LR, sigma_weight_decay=SIGMA_WD,
               grad_clip=1.0, grad_clip_scope="group")
    return _build(cfg)


@pytest.fixture(scope="module")
def adamw() -> Dict[str, Any]:
    r"""The same model under the AdamW table step and per-role clipping."""
    cfg = _cfg(m_table_update_mode="adamw", decode_mode="linear", m_p_sigma_lr=SIGMA_LR,
               sigma_weight_decay=SIGMA_WD, grad_clip=1.0, grad_clip_scope="role")
    return _build(cfg)


def _schedule(cfg: Any):
    from vfe3.train import _floor_lr_lambdas
    return _floor_lr_lambdas([cfg.m_p_sigma_lr], cfg)[0]


# ---- F21: the decay-only reference under the Fisher table step -----------------------------------

def test_fisher_logged_step_matches_the_trainer(fisher) -> None:
    r"""The premise: ``lr_sigma`` is :math:`\eta f(s)` and the table decays at :math:`\mathrm{lr}_t\lambda`."""
    from vfe3.train import _table_fisher_learning_rates
    optimizer = fisher["optimizer"]
    group = next(g for g in optimizer.param_groups if g.get("table_fisher") == "sigma")
    logged = _table_fisher_learning_rates(optimizer)["sigma"]
    assert logged == pytest.approx(ETA * group["lr"] / group["initial_lr"], rel=1e-12)
    assert group["initial_lr"] == pytest.approx(SIGMA_LR) and group["weight_decay"] == pytest.approx(SIGMA_WD)


@pytest.mark.parametrize("checkpoint", [True, False])
def test_fisher_lr_sigma_is_rescaled_to_the_decay_rate(tmp_path: Path, fisher, checkpoint: bool) -> None:
    cfg = fisher["cfg"]
    f = _schedule(cfg)
    steps = [0, 5, 10, 15, 20]
    logged = {"step": steps, "lr_sigma": [ETA * f(s) for s in steps]}            # train.py lr_sigma, item 37
    run = pr.load_run(_write_run(tmp_path, f"fisher_{checkpoint}", fisher, weights=False,
                                 checkpoint=checkpoint, metrics=logged))
    lrs, info = pf.sigma_schedule(run, 20)
    assert info["lr_crosscheck"] == "ok"
    assert info["lr_logged_as"].startswith("Fisher step")
    np.testing.assert_allclose(lrs, [SIGMA_LR * f(k) for k in range(20)], rtol=1e-12)
    assert info["weight_decay"] == pytest.approx(SIGMA_WD)


def test_fisher_runs_logged_with_the_adamw_rate_still_cross_check(tmp_path: Path, fisher) -> None:
    r"""A fisher run logged before item 37 recorded the AdamW rate; it matches that convention."""
    cfg = fisher["cfg"]
    f = _schedule(cfg)
    steps = [0, 5, 10, 15, 20]
    logged = {"step": steps, "lr_sigma": [SIGMA_LR * f(s) for s in steps]}
    run = pr.load_run(_write_run(tmp_path, "pre37", fisher, weights=False, checkpoint=True, metrics=logged))
    lrs, info = pf.sigma_schedule(run, 20)
    assert info["lr_crosscheck"] == "ok" and info["lr_logged_as"].startswith("AdamW rate")
    np.testing.assert_allclose(lrs, [SIGMA_LR * f(k) for k in range(20)], rtol=1e-12)


def test_fisher_mismatch_fallback_rescales_the_logged_step(tmp_path: Path, fisher) -> None:
    r"""On a schedule mismatch the logged Fisher step is converted by :math:`\mathrm{lr}_0/\eta`, not used raw."""
    steps = [0, 5, 10, 15, 20]
    logged = {"step": steps, "lr_sigma": [0.5 * ETA] * 5}                     # a constant factor 0.5
    run = pr.load_run(_write_run(tmp_path, "mismatch", fisher, weights=False, checkpoint=True, metrics=logged))
    lrs, info = pf.sigma_schedule(run, 20)
    assert info["lr_crosscheck"].startswith("mismatch")
    np.testing.assert_allclose(lrs, 0.5 * SIGMA_LR, rtol=1e-12)


def test_adamw_runs_keep_the_logged_rate_convention(tmp_path: Path, adamw) -> None:
    cfg = adamw["cfg"]
    f = _schedule(cfg)
    steps = [0, 5, 10, 15, 20]
    logged = {"step": steps, "lr_sigma": [SIGMA_LR * f(s) for s in steps]}
    run = pr.load_run(_write_run(tmp_path, "adamw", adamw, weights=False, checkpoint=True, metrics=logged))
    lrs, info = pf.sigma_schedule(run, 20)
    assert info["lr_crosscheck"] == "ok" and info["lr_logged_as"].startswith("AdamW rate")
    np.testing.assert_allclose(lrs, [SIGMA_LR * f(k) for k in range(20)], rtol=1e-12)
    wrong = {"step": steps, "lr_sigma": [0.5] * 5}
    run = pr.load_run(_write_run(tmp_path, "adamw_wrong", adamw, weights=False, checkpoint=True, metrics=wrong))
    lrs, info = pf.sigma_schedule(run, 20)
    assert info["lr_crosscheck"].startswith("mismatch") and np.allclose(lrs, 0.5)


# ---- F22: the Optimization field of the methods contract -----------------------------------------

def test_optimizer_groups_surface_the_table_fisher_marker(tmp_path: Path, fisher) -> None:
    run = pr.load_run(_write_run(tmp_path, "groups", fisher, weights=False, checkpoint=True))
    rows = pr.optimizer_groups(run)["groups"]
    marked = {row["table_fisher"]: row["names"] for row in rows if row.get("table_fisher")}
    assert marked == {"mu": ["prior_bank.mu_embed"], "sigma": ["prior_bank.sigma_log_embed"]}


def test_methods_contract_reports_the_fisher_table_step_and_group_clip(tmp_path: Path, fisher) -> None:
    run = pr.load_run(_write_run(tmp_path, "contract", fisher, weights=False, checkpoint=True))
    text = pt.render_items(pt.methods_contract_items(run, "final")["Optimization"], False)
    assert "m_table_update_mode=fisher" in text
    assert f"m_table_fisher_lr={ETA:g}" in text and f"m_table_fisher_trust_radius={RADIUS:g}" in text
    sigma_line = next(part for part in text.split("; ")
                      if part.strip("`").startswith("prior_bank.sigma_log_embed:"))
    assert "Fisher" in sigma_line and "lr_t/lr_0" in sigma_line
    phi_line = next(part for part in text.split("; ") if part.strip("`").startswith("prior_bank.phi_embed:"))
    assert "Fisher" not in phi_line
    clip = text.split("; gradient clipping ", 1)[1]
    assert clip.startswith("per optimizer group") and "per optimizer role" not in text
    n_clipped = sum(1 for g in fisher["optimizer"].param_groups if not g.get("table_fisher"))
    assert f"({n_clipped} groups)" in clip
    exempt = text.split("not clipped", 1)[1]
    assert "prior_bank.mu_embed" in exempt and "prior_bank.sigma_log_embed" in exempt


def test_methods_contract_adamw_role_clip_omits_inactive_fisher_settings(tmp_path: Path, adamw) -> None:
    run = pr.load_run(_write_run(tmp_path, "contract_adamw", adamw, weights=False, checkpoint=True))
    text = pt.render_items(pt.methods_contract_items(run, "final")["Optimization"], False)
    assert "m_table_update_mode=adamw" in text
    assert "m_table_fisher_lr" not in text and "m_table_fisher_trust_radius" not in text
    assert "Fisher" not in text and "not clipped" not in text
    assert "per optimizer role" in text and "(roles: mu, phi, sigma)" in text


def test_methods_contract_fisher_without_a_checkpoint_says_the_tables_are_unclipped() -> None:
    run = RunRecord(run_dir=Path("no_such_run"), config_raw={
        "m_table_update_mode": "fisher", "m_table_fisher_lr": 0.1, "m_table_fisher_trust_radius": 0.5,
        "grad_clip": 1.0, "grad_clip_scope": "group"})
    text = pt.render_items(pt.methods_contract_items(run, "final")["Optimization"], False)
    assert "per optimizer group" in text and "not clipped" in text
    assert "Fisher" in text and "m_table_fisher_lr=0.1" in text


