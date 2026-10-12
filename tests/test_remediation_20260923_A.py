"""Group A remediation of the 2026-09-23 deep audit: config activity, launchers, sweeps.

Every model here is tiny (V=11, K=4, H=2, N=6) and runs on CPU. Inertness claims are checked the
way the audit measured them: the loss and every parameter gradient must stay ``torch.equal`` when
only the marked knob changes, on a copy of the launcher route shrunk to tiny dimensions.
"""
from tests.profiles import config_values, test_config
from dataclasses import fields
import logging
import math
import warnings
from typing import Any, Dict

import pytest
import torch

import ablation
import parameter_optimization
import train_vfe3
from vfe3.config import ConfigNotice, VFE3Config, serialize_config
from vfe3.model.model import VFEModel

V, K, H, N, B = 11, 4, 2, 6, 2


# The route these rules were derived on: the launcher as audited on 2026-09-23. Pinned so the
# tests do not track later edits to the user-editable launcher dict.
_AUDITED_ROUTE = dict(
    n_layers=2, family="gaussian_full", decode_mode="full_chunked", untie_decode_bank=True,
    attention_mode="projected_predictive",
    attention_residual=True, predictive_covariance="passthrough",
    gauge_transport="on", pos_phi="learned", use_block_mlp=True,
    block_mlp_mode="coordinate", block_mlp_covariance="passthrough", block_mlp_expansion=2,
    decode_coordinates="local", decode_unigram_prior=True,
    decode_tau=0.01, lambda_beta=1.0, lambda_alpha_mode="state_dependent",
    transport_chart_max_norm=18.0, decode_av_precision="fp32",
    decode_ranking_fp64_escalation=False,
    skip_belief_sigma_update=False,
)


def _launcher(**overrides: Any) -> Dict[str, Any]:
    """Explicit tiny operating point for the historical launcher route."""
    cfg = config_values()
    cfg.update(_AUDITED_ROUTE)
    cfg.update(vocab_size=V, embed_dim=K, n_heads=H, max_seq_len=N, kl_max=8 * K,
               amp_dtype=None, batch_size=B, eval_stride=None, attention_window=N)
    cfg.update(overrides)
    return cfg


def _cfg(cfg_dict: Dict[str, Any]) -> VFE3Config:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return test_config(**cfg_dict)


def _loss_and_grads(cfg_dict: Dict[str, Any], dtype: torch.dtype = torch.float32):
    cfg = _cfg(cfg_dict)
    torch.manual_seed(0)
    model = VFEModel(cfg).to(dtype)
    if getattr(model.prior_bank, "decode_unigram_prior", False):
        counts = torch.randint(1, 50, (cfg.vocab_size,), generator=torch.Generator().manual_seed(1))
        model.prior_bank.set_unigram_log_prior(counts)
    g = torch.Generator().manual_seed(7)
    ids = torch.randint(0, cfg.vocab_size, (B, N), generator=g)
    tgt = torch.randint(0, cfg.vocab_size, (B, N), generator=g)
    # estep_grad_out forces the capture path, under which the stack computes the prior handoff.
    _, loss, _ = model(ids, tgt, estep_grad_out={})
    loss.backward()
    return model, ids, loss.detach(), {name: p.grad for name, p in model.named_parameters()}


def _assert_bit_inert(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    _, _, loss_a, grads_a = _loss_and_grads(base)
    _, _, loss_b, grads_b = _loss_and_grads({**base, **override})
    assert torch.equal(loss_a, loss_b), override
    for name, grad in grads_a.items():
        other = grads_b[name]
        assert (grad is None) == (other is None), name
        if grad is not None:
            assert torch.equal(grad, other), (override, name)


# --------------------------------------------------------------------------- #3 structural rules

_PREDICTIVE_INERT = [
    {"lambda_beta": 5.0},
    {"sigma_max": 1e-3},
    {"mu_trust_cholesky_rounds": 0},
    {"prior_handoff_rho": 0.0},
    {"prior_handoff_sigma": 1.0},
    {"attention_window": 1},
]


@pytest.mark.parametrize("override", _PREDICTIVE_INERT, ids=lambda o: next(iter(o)))
def test_predictive_route_knob_is_marked_and_bit_inert(override: Dict[str, Any]) -> None:
    base = _launcher()
    assert next(iter(override)) in _cfg(base).inactive_fields()
    _assert_bit_inert(base, override)


def _params_after_two_steps(cfg_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    from vfe3.train import build_optimizer
    cfg = _cfg(cfg_dict)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    model.prior_bank.set_unigram_log_prior(
        torch.randint(1, 50, (V,), generator=torch.Generator().manual_seed(1)))
    optimizer = build_optimizer(model, cfg)
    g = torch.Generator().manual_seed(7)
    ids = torch.randint(0, V, (B, N), generator=g)
    tgt = torch.randint(0, V, (B, N), generator=g)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        model(ids, tgt)[1].backward()
        optimizer.step()
    return {name: p.detach().clone() for name, p in model.named_parameters()}


@pytest.mark.parametrize("gauge_transport", ["off", "frozen"])
def test_frozen_frame_optimizer_knobs_are_marked_and_inert(gauge_transport: str) -> None:
    base = _launcher(gauge_transport=gauge_transport)
    assert {"m_pos_phi_lr", "phi_weight_decay"} <= set(_cfg(base).inactive_fields())
    assert "phi_weight_decay" not in _cfg(_launcher()).inactive_fields()
    reference = _params_after_two_steps(base)
    changed = _params_after_two_steps({**base, "m_pos_phi_lr": 0.5, "phi_weight_decay": 0.9})
    assert all(torch.equal(reference[name], changed[name]) for name in reference)


def test_decode_notices_only_report_active_explicit_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    # Notice policy compares dataclass declarations, so give this test its own declarations too.
    values = config_values(untie_decode_bank=False)
    for field in fields(VFE3Config):
        monkeypatch.setattr(field, "default", values[field.name])
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConfigNotice)
        test_config(decode_mode="linear", untie_decode_bank=False, decode_tau=1.0)
        test_config(decode_mode="diagonal_chunked", untie_decode_bank=False)
    inactive = test_config(decode_mode="linear", untie_decode_bank=False).inactive_fields()
    assert "untie_decode_bank" in inactive
    assert "decode_tau" in inactive
    with pytest.warns(ConfigNotice, match="decode_tau"):
        test_config(decode_mode="linear", decode_tau=0.5)


def test_launcher_route_inactive_report() -> None:
    inactive = _cfg(_launcher()).inactive_fields()
    for name in ("family", "lambda_beta", "sigma_max",
                 "mu_trust_cholesky_rounds", "attention_window", "prior_handoff_rho",
                 "prior_handoff_sigma"):
        assert name in inactive, name
    for name in ("decode_tau", "kappa_beta", "kl_max", "untie_decode_bank"):
        assert name not in inactive, name


# --------------------------------------------------------------------------- #1 family equivalence

def _fp64_pair(full_cfg: Dict[str, Any]):
    diag_cfg = {**full_cfg, "family": "gaussian_diagonal", "decode_mode": "diagonal_chunked"}
    torch.manual_seed(0)
    full = VFEModel(_cfg(full_cfg)).double()
    torch.manual_seed(0)
    diag = VFEModel(_cfg(diag_cfg)).double()
    with torch.no_grad():
        g = torch.Generator().manual_seed(3)
        for name, p in full.named_parameters():
            if "sigma_log" in name:                     # spread the constant sigma_init tables
                p.add_(0.5 * torch.randn(p.shape, generator=g, dtype=p.dtype))
    diag.load_state_dict(full.state_dict(), strict=True)
    counts = torch.randint(1, 50, (V,), generator=torch.Generator().manual_seed(1))
    for model in (full, diag):
        model.prior_bank.set_unigram_log_prior(counts)
    g = torch.Generator().manual_seed(7)
    ids = torch.randint(0, V, (B, N), generator=g)
    tgt = torch.randint(0, V, (B, N), generator=g)
    return full, diag, ids, tgt


def _off_diagonal(sigma: torch.Tensor) -> float:
    return float((sigma - torch.diag_embed(torch.diagonal(sigma, dim1=-2, dim2=-1))).abs().max())


def test_full_family_is_value_identical_to_diagonal_on_launcher_route() -> None:
    base = _launcher()
    assert "family" in _cfg(base).inactive_fields()
    full, diag, ids, tgt = _fp64_pair(base)
    with torch.no_grad():
        belief, _ = full.forward_beliefs(ids, return_logits=False)
    assert _off_diagonal(belief.sigma) == 0.0
    _, loss_full, _ = full(ids, tgt)
    _, loss_diag, _ = diag(ids, tgt)
    # The two families decode through different kernels (full against diagonal chunked). Bitwise equality held
    # only while the decode rounded float64 beliefs to float32; in float64 they agree to roundoff (audit
    # 2026-10-10, item 3).
    assert math.isclose(loss_full.item(), loss_diag.item(), rel_tol=1e-13, abs_tol=0.0)


@pytest.mark.parametrize("override", [
    {"block_mlp_covariance": "delta_full", "norm_type_block": "none"},
    {"predictive_covariance": "moment_matched"},
], ids=lambda o: next(iter(o)))
def test_family_stays_live_where_a_seam_densifies_or_diagonal_is_not_admitted(
    override: Dict[str, Any],
) -> None:
    assert "family" not in _cfg(_launcher(**override)).inactive_fields()


def test_passthrough_gauge_gate_keeps_family_value_identical() -> None:
    # The gauge gate admits the diagonal family (2026-09-28): on a diagonal Sigma its invariant
    # mu_b^T Sigma_bb^{-1} mu_b is sum_k mu_k^2 / sigma_k, exact up to the Cholesky solve's roundoff.
    base = _launcher(block_mlp_mode="gauge_gate", predictive_covariance="passthrough",
                     decode_coordinates="local")
    assert "family" in _cfg(base).inactive_fields()
    full, diag, ids, tgt = _fp64_pair(base)
    with torch.no_grad():
        belief, _ = full.forward_beliefs(ids, return_logits=False)
    assert _off_diagonal(belief.sigma) == 0.0
    _, loss_full, _ = full(ids, tgt)
    _, loss_diag, _ = diag(ids, tgt)
    loss_full.backward()
    loss_diag.backward()
    torch.testing.assert_close(loss_full, loss_diag, rtol=1e-12, atol=0.0)
    grads_diag = dict(diag.named_parameters())
    for name, p in full.named_parameters():
        other = grads_diag[name].grad
        assert (p.grad is None) == (other is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, other, rtol=1e-12, atol=1e-14,
                                       msg=lambda m, name=name: f"{name}: {m}")


def test_delta_full_densifies_the_stored_covariance() -> None:
    cfg = _cfg(_launcher(block_mlp_covariance="delta_full", norm_type_block="none"))
    torch.manual_seed(0)
    model = VFEModel(cfg).double()
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.1 * torch.randn(p.shape, generator=g, dtype=p.dtype))
        belief, _ = model.forward_beliefs(torch.randint(0, V, (B, N), generator=g),
                                          return_logits=False)
    assert _off_diagonal(belief.sigma) > 1e-3


# --------------------------------------------------------------------------- #8 capacity control


# --------------------------------------------------------------------------- #26, low validation


@pytest.mark.parametrize("value", [-1.0, math.nan, math.inf])
def test_mu_weight_decay_is_validated(value: float) -> None:
    with pytest.raises(ValueError, match="mu_weight_decay"):
        test_config(mu_weight_decay=value)


# --------------------------------------------------------------------------- launcher (#23, #24, low)

def test_resume_from_rejected_with_more_than_one_seed() -> None:
    with pytest.raises(ValueError, match="resume_from"):
        train_vfe3._resolve_seeds({"resume_from": "ckpt.pt"}, seeds=[6, 7], num_runs=2)
    assert train_vfe3._resolve_seeds({"resume_from": "ckpt.pt"}, seeds=[6, 7], num_runs=1) == [6]


def test_seeds_come_only_from_SEEDS() -> None:
    assert "seed" not in train_vfe3.config and "seed" not in ablation.BASELINE_CONFIG
    assert train_vfe3._resolve_seeds({}, seeds=[6, 54, 271], num_runs=2) == [6, 54]
    with pytest.raises(ValueError, match="must not set 'seed'"):
        train_vfe3._resolve_seeds({"seed": 6}, seeds=[6], num_runs=1)
    with pytest.raises(ValueError, match="at least NUM_RUNS"):
        train_vfe3._resolve_seeds({}, seeds=[], num_runs=1)


@pytest.mark.parametrize("data_seed,expected", [(None, 23), (3, 3)])
def test_train_loader_always_has_a_resumable_generator(
    monkeypatch: pytest.MonkeyPatch, data_seed: Any, expected: int,
) -> None:
    captured: Dict[str, Any] = {}

    def fake_make_dataloader(dataset, split, seq_len, batch_size, **kw):
        captured[split] = kw["generator"]
        return ("loader", split)

    monkeypatch.setattr(train_vfe3, "make_dataloader", fake_make_dataloader)
    monkeypatch.setattr(train_vfe3, "DATA_SEED", data_seed)
    cfg = test_config(seed=23)
    for split in ("train", "validation"):
        train_vfe3._select_loader("wikitext-103", cfg, split=split)
    assert captured["train"].initial_seed() == expected
    assert captured["validation"] is None


def test_banner_probe_keeps_the_unigram_warning_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    import vfe3.model.prior_bank as prior_bank
    monkeypatch.setattr(prior_bank, "_WARNED_UNIGRAM_UNSET", False)
    cfg = _cfg(_launcher())
    model = VFEModel(cfg)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        train_vfe3._banner(model, cfg, "wikitext-103", "cpu", 1)
    assert not any("unigram_log_prior table is unset" in str(w.message) for w in caught)
    assert prior_bank._WARNED_UNIGRAM_UNSET is False


def test_run_once_prints_inactive_launcher_settings(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    class Stop(Exception):
        pass

    def stop(*args: Any, **kwargs: Any) -> None:
        raise Stop

    monkeypatch.setattr(train_vfe3, "config", _launcher())
    monkeypatch.setattr(train_vfe3, "DEVICE", "cpu")
    monkeypatch.setattr(train_vfe3, "DATASET", "wikitext-103")
    monkeypatch.setattr(train_vfe3, "seed_everything", stop)       # halt right after the report
    with caplog.at_level(logging.INFO), pytest.raises(Stop):
        train_vfe3._run_once(6, logging.getLogger("inactive-report"))
    assert "lambda_beta=1.0" in caplog.text and "family='gaussian_full'" in caplog.text
    assert "kappa_beta" not in caplog.text


# --------------------------------------------------------------------------- sweeps and leaderboards

def test_inert_sweep_is_rejected_when_selected(monkeypatch: pytest.MonkeyPatch) -> None:
    baseline = serialize_config(_cfg(_launcher()))
    monkeypatch.setattr(ablation, "_resolved_sweep_baseline", lambda: dict(baseline))
    # The registered sigma_max sweep requires its sigma E-step; without it the ceiling is inert on this baseline.
    monkeypatch.setitem(ablation.SWEEPS, "sigma_max_unrouted", {"param": "sigma_max", "values": [1, 100]})
    # The registered lambda_beta sweep requires projected_gradient; without it the weight is inert on this baseline.
    monkeypatch.setitem(ablation.SWEEPS, "lambda_beta_unrouted", {"param": "lambda_beta", "values": [0.5, 1.0]})
    for name in ("lambda_beta_unrouted", "sigma_max_unrouted"):
        with pytest.raises(ValueError, match=f"{name}.*INERT on EVERY"):
            ablation.validate_sweeps([name], require_construction=True)
    ablation.validate_sweeps(list(ablation.SWEEPS), require_construction=False)   # registry stays valid


def test_single_seed_arm_has_unknown_spread_and_no_error_bar(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    rows = [{"label": "a__s0", "primary_val_ppl": "30.0"},
            {"label": "a__s1", "primary_val_ppl": "32.0"},
            {"label": "b", "primary_val_ppl": "40.0"}]
    saved = []
    monkeypatch.setattr(ablation, "_read_sweep_csv", lambda _dir: rows)
    monkeypatch.setattr(ablation, "_save_ablation_figure",
                        lambda fig, out, sweep_dir, plt: saved.append(fig))
    ablation._plot_seed_aggregate(tmp_path / "sweep", tmp_path / "figures")
    (fig,) = saved
    (ax,) = fig.axes
    assert len(ax.containers) == 1 and len(ax.containers[0].lines[0].get_xdata()) == 1
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_leaderboard_ranks_complete_candidates_despite_a_failed_cell() -> None:
    candidates = [{"label": "a", "overrides": {}}, {"label": "b", "overrides": {}},
                  {"label": "c", "overrides": {}}]
    results = [{"label": "a__s0", "seed": 0, "primary_val_ppl": 30.0},
               {"label": "a__s1", "seed": 1, "primary_val_ppl": 32.0},
               {"label": "b__s0", "seed": 0, "primary_val_ppl": 10.0},
               {"label": "c__s0", "seed": 0, "primary_val_ppl": 20.0},
               {"label": "c__s1", "seed": 1, "primary_val_ppl": 22.0}]
    rows = parameter_optimization._leaderboard_rows(candidates, results, 2)
    assert [(row["label"], row["rank"], row["status"]) for row in rows] == [
        ("c", 1, "complete"), ("a", 2, "complete"), ("b", None, "incomplete")]
