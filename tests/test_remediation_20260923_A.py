"""Group A remediation of the 2026-09-23 deep audit: config activity, launchers, sweeps.

Every model here is tiny (V=11, K=4, H=2, N=6) and runs on CPU. Inertness claims are checked the
way the audit measured them: the loss and every parameter gradient must stay ``torch.equal`` when
only the marked knob changes, on a copy of the launcher route shrunk to tiny dimensions.
"""
import copy
import logging
import math
import warnings
from typing import Any, Dict

import pytest
import torch

import ablation
import parameter_optimization
import train_vfe3
from vfe3.config import ConfigNotice, VFE3Config, config_from_serialized, serialize_config
from vfe3.model.model import VFEModel

V, K, H, N, B = 11, 4, 2, 6, 2


def _launcher(**overrides: Any) -> Dict[str, Any]:
    """The train_vfe3 config dict at tiny width (every other launcher value kept)."""
    cfg = copy.deepcopy(dict(train_vfe3.config))
    cfg.update(vocab_size=V, embed_dim=K, n_heads=H, max_seq_len=N, kl_max=8 * K,
               amp_dtype=None, batch_size=B, eval_stride=None, attention_window=N)
    cfg.update(overrides)
    return cfg


def _conditional(**overrides: Any) -> Dict[str, Any]:
    return _launcher(attention_mode="projected_conditional", n_layers=1,
                     relational_output_scope="head", attention_residual=False, **overrides)


def _transported(**overrides: Any) -> Dict[str, Any]:
    base = dict(attention_mode="transported", n_layers=1, relational_output_scope="head",
                attention_residual=False, family="gaussian_diagonal",
                decode_mode="diagonal_chunked")
    return _launcher(**{**base, **overrides})


def _cfg(cfg_dict: Dict[str, Any]) -> VFE3Config:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VFE3Config(**cfg_dict)


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
    {"full_cov_kl_backend": "pairwise"},
    {"full_cov_congruence_precision": "fp64"},
    {"full_cov_congruence_certification": True},
    {"congruence_cond_escalation": True},
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


def test_conditional_route_marks_only_its_inert_knobs() -> None:
    inactive = _cfg(_conditional()).inactive_fields()
    for name in ("sigma_max", "full_cov_kl_backend", "e_q_sigma_lr", "e_sigma_q_trust",
                 "lambda_alpha", "lambda_alpha_mode", "b0", "c0"):
        assert name in inactive, name
    # Live here: lambda_beta weights the conditional coupling, the value covariance is
    # transported by congruence, and the full-covariance mean trust region factors sigma_q.
    for name in ("lambda_beta", "full_cov_congruence_precision",
                 "full_cov_congruence_certification", "congruence_cond_escalation",
                 "mu_trust_cholesky_rounds", "family"):
        assert name not in inactive, name
    _assert_bit_inert(_conditional(), {"sigma_max": 1e-3, "full_cov_kl_backend": "pairwise"})


def test_self_coupling_inert_at_one_iteration_one_layer() -> None:
    base = _transported()
    assert {"lambda_alpha", "lambda_alpha_mode"} <= set(_cfg(base).inactive_fields())
    _assert_bit_inert(base, {"lambda_alpha": 5.0})
    _assert_bit_inert(base, {"lambda_alpha_mode": "state_dependent", "b0": 0.1, "c0": 3.0})
    two_steps = _transported(n_e_steps=2)
    assert "lambda_alpha" not in _cfg(two_steps).inactive_fields()
    _, _, loss_a, _ = _loss_and_grads(two_steps)
    _, _, loss_b, _ = _loss_and_grads({**two_steps, "lambda_alpha": 5.0})
    assert not torch.equal(loss_a, loss_b)


def test_frozen_sigma_channel_is_marked_and_bit_inert() -> None:
    base = _transported(skip_belief_sigma_update=True, n_e_steps=2)
    assert {"e_q_sigma_lr", "e_sigma_q_trust", "sigma_max"} <= set(_cfg(base).inactive_fields())
    _assert_bit_inert(base, {"e_q_sigma_lr": 0.1, "e_sigma_q_trust": 50.0, "sigma_max": 1e-3})
    live = {**base, "skip_belief_sigma_update": False}
    assert "sigma_max" not in _cfg(live).inactive_fields()


@pytest.mark.parametrize("functional", ["bhattacharyya", "jeffreys", "squared_hellinger"])
def test_renyi_order_inert_under_order_free_functionals(functional: str) -> None:
    base = _transported(n_e_steps=2, decode_mode="family_chunked", divergence_family=functional,
                        renyi_order=0.5)
    assert "renyi_order" in _cfg(base).inactive_fields()
    _assert_bit_inert(base, {"renyi_order": 2.0})
    assert "renyi_order" not in _cfg({**base, "divergence_family": "renyi"}).inactive_fields()


def test_windowed_prior_keeps_attention_window_live() -> None:
    assert "attention_window" not in _cfg(
        _transported(beta_attention_prior="causal_windowed", attention_window=2)).inactive_fields()


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


def test_fp64_island_unreachable_above_chart_bound() -> None:
    cfg = _cfg(_launcher())                            # launcher: threshold 20 > chart 18
    assert "exp_fp64_norm_threshold" in cfg.inactive_fields()
    assert "exp_fp64_norm_threshold" not in _cfg(
        _launcher(exp_fp64_norm_threshold=18.0)).inactive_fields()      # equality stays reachable


def test_transported_frame_diagonal_drops_positional_channels() -> None:
    base = _transported(family="gaussian_frame_diagonal", decode_mode="family_chunked",
                        n_e_steps=2, pos_phi="none", rope_on_value=True)
    inactive = _cfg(base).inactive_fields()
    assert {"pos_rotation", "pos_phi", "rope_base", "m_pos_phi_lr"} <= set(inactive)
    _assert_bit_inert(base, {"pos_rotation": "rope", "rope_base": 2.0})


def test_decode_notices_are_default_guarded() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConfigNotice)
        VFE3Config()                                                    # untie_decode_bank default
        VFE3Config(decode_mode="diagonal_chunked")                      # decode_bias default
    assert "untie_decode_bank" in VFE3Config().inactive_fields()
    assert "decode_tau" in VFE3Config().inactive_fields()
    with pytest.warns(ConfigNotice, match="decode_tau"):
        VFE3Config(decode_tau=0.5)


def test_launcher_route_inactive_report() -> None:
    inactive = _cfg(_launcher()).inactive_fields()
    for name in ("family", "lambda_beta", "sigma_max", "full_cov_kl_backend",
                 "mu_trust_cholesky_rounds", "attention_window", "prior_handoff_rho",
                 "prior_handoff_sigma", "decode_bias", "exp_fp64_norm_threshold"):
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
    assert loss_full.item() == loss_diag.item()


@pytest.mark.parametrize("override", [
    {"block_mlp_covariance": "delta_full"},
    {"use_head_mixer": True},
    {"block_mlp_mode": "gauge_gate"},
], ids=lambda o: next(iter(o)))
def test_family_stays_live_where_a_seam_densifies_or_diagonal_is_not_admitted(
    override: Dict[str, Any],
) -> None:
    assert "family" not in _cfg(_launcher(**override)).inactive_fields()


def test_delta_full_densifies_the_stored_covariance() -> None:
    cfg = _cfg(_launcher(block_mlp_covariance="delta_full"))
    torch.manual_seed(0)
    model = VFEModel(cfg).double()
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.1 * torch.randn(p.shape, generator=g, dtype=p.dtype))
        belief, _ = model.forward_beliefs(torch.randint(0, V, (B, N), generator=g),
                                          return_logits=False)
    assert _off_diagonal(belief.sigma) > 1e-3


def test_conditional_route_keeps_family_live() -> None:
    assert "family" not in _cfg(_conditional()).inactive_fields()


# --------------------------------------------------------------------------- #8 capacity control

@pytest.mark.parametrize("route", ["projected_predictive", "projected_conditional"])
def test_additive_encoder_admitted_on_projected_routes(route: str) -> None:
    base = (_launcher() if route == "projected_predictive" else _conditional())
    base.update(encode_mode="per_token_additive", pos_phi="none")
    model, _, loss, grads = _loss_and_grads(base)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(g).all() for g in grads.values() if g is not None)
    assert grads["prior_bank.phi_embed"] is not None                # the table stays trained
    with pytest.raises(ValueError, match="per_token_additive.*positional frame"):
        VFE3Config(**{**base, "pos_phi": "learned"})


def test_additive_encoder_contamination_warns_on_transported() -> None:
    with pytest.warns(UserWarning, match="per_token_additive"):
        VFE3Config(**_transported(encode_mode="per_token_additive", pos_phi="learned"))


# --------------------------------------------------------------------------- #26, low validation

def test_legacy_dense_decode_admission_round_trips() -> None:
    cfg = VFE3Config(decode_mode="full", family="gaussian_full", oracle_unroll_grad=True,
                     _allow_legacy_dense_decode=True)
    payload = serialize_config(cfg)
    assert payload["_allow_legacy_dense_decode"] is True
    assert config_from_serialized(payload, source="round trip").decode_mode == "full"
    assert "_allow_legacy_dense_decode" not in serialize_config(VFE3Config())


@pytest.mark.parametrize("value", [-1.0, math.nan, math.inf])
def test_mu_weight_decay_is_validated(value: float) -> None:
    with pytest.raises(ValueError, match="mu_weight_decay"):
        VFE3Config(mu_weight_decay=value)


# --------------------------------------------------------------------------- launcher (#23, #24, low)

def test_resume_from_rejected_with_more_than_one_seed() -> None:
    with pytest.raises(ValueError, match="resume_from"):
        train_vfe3._resolve_seeds({"seed": 6, "resume_from": "ckpt.pt"}, seeds=[6, 7], num_runs=2)
    assert train_vfe3._resolve_seeds({"seed": 6, "resume_from": "ckpt.pt"},
                                     seeds=[6], num_runs=1) == [6]


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
    cfg = VFE3Config(seed=23)
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
    monkeypatch.setattr(train_vfe3, "seed_everything", stop)       # halt right after the report
    with caplog.at_level(logging.INFO), pytest.raises(Stop):
        train_vfe3._run_once(6, logging.getLogger("inactive-report"))
    assert "lambda_beta=1.0" in caplog.text and "family='gaussian_full'" in caplog.text
    assert "kappa_beta" not in caplog.text


# --------------------------------------------------------------------------- sweeps and leaderboards

def test_inert_sweep_is_rejected_when_selected(monkeypatch: pytest.MonkeyPatch) -> None:
    baseline = serialize_config(_cfg(_launcher()))
    monkeypatch.setattr(ablation, "_resolved_sweep_baseline", lambda: dict(baseline))
    for name in ("lambda_beta", "sigma_max"):
        with pytest.raises(ValueError, match=f"{name}.*INERT on EVERY"):
            ablation.validate_sweeps([name], require_construction=True)
    ablation.validate_sweeps(list(ablation.SWEEPS), require_construction=False)   # registry stays valid


def test_lambda_alpha_sweeps_require_two_e_steps() -> None:
    assert ablation.SWEEPS["lambda_alpha"]["requires"]["n_e_steps"] == 2
    spec = parameter_optimization.SEARCH_CONFIG["parameters"]["lambda_alpha"]
    assert spec["requires"]["n_e_steps"] == 2


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
