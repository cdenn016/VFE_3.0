"""Round-2 remediation of the October 9, 2026 audit: training telemetry and analysis readers.

Items 9 (per-head, reference-aware dead-router statistics), 24 (``belief_cond_median_layer_max`` export),
25 (``sigma_chol_embed`` telemetry), 26 (per-element step in learning-rate units), 34 (fixed training sample
under per-epoch offsets), 35 (run totals across resume), 40 (autocast in replay records), 5 (M4 and klcap
under ``row_min``), 23 and 44 (figures and readout guard fractions), 33 (``block_cholesky`` readers).
Tiny CPU models only.
"""
import logging
import math

import numpy as np

import pytest
import torch

import vfe3.train as T
from tests.profiles import test_config
from vfe3.model.model import VFEModel


# --------------------------------------------------------------------------- item 9: per-head router statistics

def test_router_head_statistics_separate_self_lock_from_clamped_rows() -> None:
    N = 3
    support = torch.ones(1, 2, N, N, dtype=torch.bool).tril()               # causal (1, H, N, N)
    beta = torch.zeros(1, 2, N, N)
    beta[0, 0] = torch.eye(N)                                               # head 0: one-hot self route
    beta[0, 1] = support[0, 1].float() / support[0, 1].float().sum(-1, keepdim=True)   # head 1: uniform
    score = torch.zeros(1, 2, N, N)
    score[0, 0] = 5.0 * (1 - torch.eye(N))                                  # row_min gaps: off-self capped
    score[0, 1] = 5.0                                                       # absolute: every entry capped
    stats = VFEModel._router_head_statistics(score, beta, support, cap_level=5.0)
    # scored rows are 1 and 2; head 0 caps 3 of the 6 support entries, head 1 all 6
    torch.testing.assert_close(stats["klmax_frac"], torch.tensor([3 / 6, 1.0], dtype=torch.float64))
    torch.testing.assert_close(stats["clamped_row_frac"], torch.tensor([0.0, 1.0], dtype=torch.float64))
    torch.testing.assert_close(stats["self_weight"], torch.tensor([1.0, (1 / 2 + 1 / 3) / 2], dtype=torch.float64))
    torch.testing.assert_close(stats["self_argmax_frac"], torch.tensor([1.0, 1.0], dtype=torch.float64))


def test_router_head_statistics_without_self_support_are_nan() -> None:
    support = torch.ones(1, 1, 3, 3, dtype=torch.bool).tril(-1)             # strictly causal: no self
    support[..., 0, 0] = True
    beta = support.float() / support.float().sum(-1, keepdim=True)
    stats = VFEModel._router_head_statistics(torch.zeros_like(beta), beta, support, cap_level=1.0)
    assert math.isnan(float(stats["self_weight"][0])) and math.isnan(float(stats["self_argmax_frac"][0]))


def test_diagnostics_report_per_head_maxima_over_layers() -> None:
    torch.manual_seed(0)
    cfg = test_config(vocab_size=12, embed_dim=4, n_heads=2, max_seq_len=6, n_layers=3,
                      n_e_steps=1, e_q_mu_lr=0.05, prior_handoff_rho=0.5, kl_max=0.5)
    model = VFEModel(cfg)
    tokens = torch.randint(0, 12, (2, 6), generator=torch.Generator().manual_seed(1))
    snapshot = model.build_diagnostic_snapshot(tokens, return_logits=False)
    d = model.diagnostics(tokens, snapshot=snapshot)
    from vfe3.posthoc.klcap import causal_support
    table = {key: [] for key in ("klmax_frac", "clamped_row_frac", "self_weight", "self_argmax_frac")}
    for layer in range(cfg.n_layers):
        cap = model._layer_capture(snapshot, layer)
        beta, score = cap["beta_pre"], cap["score_energy_pre"]
        stats = VFEModel._router_head_statistics(score, beta, causal_support(beta, snapshot.log_prior), cfg.kl_max)
        for key in table:
            table[key].append(stats[key])
    for key, column in (("klmax_frac", "relational_score_klmax_frac_head_max"),
                        ("clamped_row_frac", "relational_clamped_row_frac_head_max"),
                        ("self_weight", "relational_self_weight_head_max"),
                        ("self_argmax_frac", "relational_self_argmax_frac_head_max")):
        stacked = torch.stack(table[key])                                   # (L, H)
        assert d[column] == pytest.approx(float(stacked.max()))
    weights = torch.stack(table["self_weight"])
    flat = int(weights.reshape(-1).argmax())
    assert (d["relational_self_lock_argmax_layer"], d["relational_self_lock_argmax_head"]) == (
        float(flat // weights.shape[1]), float(flat % weights.shape[1]))
    # the per-head maximum bounds the pooled per-layer maximum
    assert d["relational_score_klmax_frac_head_max"] >= d["relational_score_klmax_frac_layer_max"] - 1e-12
    val_keys = set(T._VAL_RELATIONAL_DIAG_KEYS)
    for key in ("relational_score_klmax_frac_head_max", "relational_clamped_row_frac_head_max",
                "relational_clamped_row_frac_argmax_layer", "relational_clamped_row_frac_argmax_head",
                "relational_self_weight_head_max", "relational_self_argmax_frac_head_max",
                "relational_self_lock_argmax_layer", "relational_self_lock_argmax_head",
                "belief_cond_median_layer_max"):
        assert key in d and f"val_{key}" in val_keys


@pytest.mark.parametrize("reference,tied", [("absolute", False), ("row_min", False), ("absolute", True)])
def test_dead_router_warning_reads_both_signals_under_every_reference(
        reference: str, tied: bool, caplog: pytest.LogCaptureFixture) -> None:
    # audit 2026-10-10 round 2, item 7: the self weight and the clamped rows are read under every reference
    logger = logging.getLogger("dead-router-reference")
    # a self-locked head (99.68 layer 1 head 0) and a head with every scored row clamped
    diagnostics = {"relational_clamped_row_frac_head_max": 0.8, "relational_clamped_row_frac_argmax_layer": 2,
                   "relational_clamped_row_frac_argmax_head": 3,
                   "relational_self_weight_head_max": 0.99, "relational_self_lock_argmax_layer": 1,
                   "relational_self_lock_argmax_head": 0}
    warned: set = set()
    with caplog.at_level(logging.WARNING, logger="dead-router-reference"):
        T._warn_dead_router(diagnostics, warned, logger, reference=reference, tied=tied)
        T._warn_dead_router(diagnostics, warned, logger, reference=reference, tied=tied)   # once only
    messages = {r.getMessage().split(":")[0].strip(): r.getMessage() for r in caplog.records}
    assert len(caplog.records) == 2
    assert "fully clamped" in messages["router layer 2 head 3"]
    assert "one-hot" in messages["router layer 1 head 0"] and "softmax saturates" in messages["router layer 1 head 0"]


def test_row_min_does_not_call_a_high_cap_fraction_uniform(caplog: pytest.LogCaptureFixture) -> None:
    # 99.68: layer 1 caps 0.53 to 0.64 of its gaps under row_min, but no head is self-locked above threshold
    logger = logging.getLogger("dead-router-row-min")
    with caplog.at_level(logging.WARNING, logger="dead-router-row-min"):
        T._warn_dead_router({"relational_score_klmax_frac_layer_max": 0.64,
                             "relational_score_klmax_frac_head_max": 0.7,
                             "relational_clamped_row_frac_head_max": 0.0,
                             "relational_self_weight_head_max": 0.6}, set(), logger, reference="row_min")
    assert not caplog.records


# --------------------------------------------------------------------------- item 25: block_cholesky telemetry

def _cholesky_model() -> VFEModel:
    from tests.profiles import tiny_config
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = tiny_config(embed_dim=6, n_heads=2, vocab_size=9, max_seq_len=5, n_layers=1, family="gaussian_block",
                          decode_mode="full_chunked", untie_decode_bank=True, block_coordinates="reference",
                          decode_coordinates="reference", norm_type_block="layernorm", layernorm_affine=True,
                          use_block_mlp=True, block_mlp_mode="coordinate", content_link_chart="exponential",
                          pos_phi="none", gauge_transport="off", prior_covariance_table="block_cholesky")
        torch.manual_seed(0)
        return VFEModel(cfg)


def test_sigma_chol_embed_is_in_the_sigma_group() -> None:
    assert T._parameter_group("prior_bank.sigma_chol_embed") == "sigma_log_embed"
    assert T._parameter_group("_orig_mod.prior_bank.sigma_chol_embed") == "sigma_log_embed"


def test_sigma_log_statistics_read_the_block_cholesky_log_variances(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _cholesky_model()
    with torch.no_grad():
        model.prior_bank.sigma_chol_embed.add_(0.3 * torch.randn_like(model.prior_bank.sigma_chol_embed))
        log_variance = model.prior_bank._prior_covariance_blocks().diagonal(dim1=-2, dim2=-1).double().log()
    monkeypatch.setattr(T, "SIGMA_LOG_CHUNK_TYPES", 4)                      # three chunks over V = 9
    stats = T._sigma_log_statistics(model)
    assert stats["sigma_log_mean"] == pytest.approx(float(log_variance.mean()), rel=1e-10)
    assert stats["sigma_log_sd"] == pytest.approx(float(log_variance.std(correction=0)), rel=1e-8)


# --------------------------------------------------------------------------- item 26: per-element step / lr

def test_update_rms_per_lr_is_the_rms_step_in_learning_rate_units() -> None:
    cfg = test_config(vocab_size=7, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=1)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    optimizer = T.build_optimizer(model, cfg)
    before = T._parameter_snapshot(model)
    rates = T._parameter_learning_rates(model, optimizer)
    with torch.no_grad():
        model.prior_bank.mu_embed.add_(0.5 * rates["prior_bank.mu_embed"])          # every element moves 0.5 lr
        step = torch.zeros_like(model.prior_bank.sigma_log_embed)
        step[0, 0] = 2.0 * rates["prior_bank.sigma_log_embed"]                   # one element moves 2 lr
        model.prior_bank.sigma_log_embed.add_(step)
    out = T._group_update_statistics(model, before, learning_rates=rates)
    assert out["update_rms_per_lr_group_mu_embed"] == pytest.approx(0.5, rel=1e-5)
    n_sigma = model.prior_bank.sigma_log_embed.numel()
    assert out["update_rms_per_lr_group_sigma_log_embed"] == pytest.approx(2.0 / math.sqrt(n_sigma), rel=1e-5)
    assert "update_ratio_group_sigma_log_embed" in out                           # the existing column is kept
    assert set(T._STEP_LR_COLUMNS) <= set(out)
    assert all(math.isnan(v) for k, v in T._group_update_statistics(model, before).items()
               if k in T._STEP_LR_COLUMNS)                                       # no rates: NaN


# --------------------------------------------------------------------------- item 34: fixed training sample

def test_fixed_training_sample_is_the_same_under_per_epoch_offsets() -> None:
    from torch.utils.data import DataLoader
    from vfe3.data.datasets import TokenWindows
    tokens = torch.arange(201) % 11
    cfg = test_config(vocab_size=11, embed_dim=4, n_heads=2, max_seq_len=8, eval_stride=8)
    val = DataLoader(TokenWindows(torch.arange(60) % 11, 8, pad_final=True), batch_size=2)
    fixed = TokenWindows(tokens, 8)
    offset = TokenWindows(tokens, 8)
    offset.use_per_epoch_offsets()
    assert (len(fixed), len(offset)) == (25, 24)                              # T = 201, L = 8
    offset.draw_window_offset(torch.Generator().manual_seed(3))
    sample_fixed, _ = T._build_fixed_train_sample(DataLoader(fixed, batch_size=2), val, cfg)
    sample_offset, _ = T._build_fixed_train_sample(DataLoader(offset, batch_size=2), val, cfg)
    assert sample_fixed.indices == sample_offset.indices
    assert len(sample_offset.dataset) == len(fixed) and sample_offset.dataset.window_offset_mode == "fixed"
    for index in sample_fixed.indices:
        assert torch.equal(sample_fixed.dataset[index][0], sample_offset.dataset[index][0])
    assert offset.window_offset_mode == "per_epoch_random" and len(offset) == 24   # the loader's view is untouched


# --------------------------------------------------------------------------- item 35: run totals across resume

def _carry_run_totals(value: object) -> object:
    r"""``_checkpoint_numerics_counters`` extended to the run totals (the run_artifacts side of item 35):
    frame-projection keys all or none, plus every non-negative integer of ``RUN_TOTAL_COUNTER_KEYS``."""
    import vfe3.run_artifacts as ra
    if not isinstance(value, dict):
        return None
    frame = {key: value.get(key) for key in ra._CHECKPOINT_NUMERICS_KEYS}
    out = frame if all(type(count) is int and count >= 0 for count in frame.values()) else {}
    out.update({key: value[key] for key in T.RUN_TOTAL_COUNTER_KEYS
                if type(value.get(key)) is int and value[key] >= 0})
    return out or None


def _resume_with_constant_counters(tmp_path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    import tests.test_training_counters_20260928 as tc
    import vfe3.run_artifacts as ra
    monkeypatch.setattr(T, "source_recovery_elements", lambda: 3)            # per-process count of 3
    monkeypatch.setattr(T, "nonfinite_tangent_elements", lambda: 2)
    cfg = tc._cfg(max_steps=6, checkpoint_interval=3, m_phi_lr=0.05)
    first_model = tc._model(cfg)
    first = ra.RunArtifacts(tmp_path / "first", cfg, first_model)
    tc._train(first_model, cfg, tc._loaders(cfg, shuffle=False), artifacts=first, log_interval=3, n_steps=3)
    checkpoint = tmp_path / "first" / "checkpoints" / "step_3.pt"
    resumed_model = tc._model(cfg, seed=99)
    resumed = ra.RunArtifacts(tmp_path / "resumed", cfg, resumed_model)
    tc._train(resumed_model, cfg, tc._loaders(cfg, shuffle=False), artifacts=resumed, log_interval=3,
              resume_from=checkpoint)
    return checkpoint, resumed, tc._rows(tmp_path / "resumed" / "metrics.csv")


def test_run_total_counters_continue_across_resume(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vfe3.run_artifacts as ra
    monkeypatch.setattr(ra, "_checkpoint_numerics_counters", _carry_run_totals)
    checkpoint, resumed, rows = _resume_with_constant_counters(tmp_path, monkeypatch)
    carried = torch.load(checkpoint, weights_only=True)["numerics_counters"]
    assert carried["source_recovery_elements"] == 3 and carried["nonfinite_tangent_elements"] == 2
    assert carried["frame_projection_since_step"] == 0                       # frame counts still carried
    assert resumed.realized_updates["source_recovery_elements"] == 6        # 3 before + 3 after the resume
    assert resumed.realized_updates["nonfinite_tangent_elements"] == 4
    assert float(rows[-1]["source_recovery_elements"]) == 6.0
    assert resumed.numerics_counters_state["source_recovery_elements"] == 6


def test_run_artifacts_checkpoint_carries_the_run_totals(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, resumed, _ = _resume_with_constant_counters(tmp_path, monkeypatch)
    assert resumed.realized_updates["source_recovery_elements"] == 6


# --------------------------------------------------------------------------- item 40: replay records

@pytest.mark.parametrize("amp_dtype", [None, "bf16"])
def test_precision_replay_records_carry_the_autocast_dtype(amp_dtype, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = test_config(vocab_size=7, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=1,
                      amp_dtype=amp_dtype)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    optimizer = T.build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    calls = {"n": 0}
    real_step = T.train_step

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("forced certificate failure")
        return real_step(*args, **kwargs)

    monkeypatch.setattr(T, "train_step", flaky)
    tokens = torch.randint(0, 7, (2, 4), generator=torch.Generator().manual_seed(2))
    records: list = []
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        _, replays = T._train_step_with_precision_retry(
            model, optimizer, scheduler, tokens[:, :-1], tokens[:, 1:], logger=logging.getLogger("replay-amp"),
            step=0, replay_log=records)
    finally:
        torch.set_float32_matmul_precision(previous)
    assert replays == 1 and len(records) == 1
    record = records[0]
    assert record["amp_dtype"] == amp_dtype
    assert record["autocast_active"] is (amp_dtype is not None)
    assert ("autocast_note" in record) is (amp_dtype is not None)


# --------------------------------------------------------------------------- item 5: M4 and klcap under row_min

def test_m4_decomposition_applies_the_row_min_reference() -> None:
    import tests.test_posthoc_routing_20260928 as tr
    result = tr._run(tr._model(tr._cfg(kl_max=1.0, routing_kl_max_reference="row_min")))
    assert result["status"] == "ok" and result["setup"]["kl_max_reference"] == "row_min"
    worst = result["checks_worst"]
    assert worst["identity_max_rel_error"] < 1e-12
    assert worst["model_cost_max_abs_error"] < 1e-10
    assert worst["model_logit_max_abs_error"] < 1e-10         # modulo the row constant m_i / tau
    assert worst["model_beta_max_abs_error"] < 1e-12           # an absolute cap would not reproduce beta here
    assert any(abs(row["share_within_cap"]) > 1e-3 for row in result["panel_a"])   # gaps above kl_max exist
    from vfe3.posthoc.routing import routing_table
    assert "(reference row_min)" in routing_table(result)


def test_logit_components_cap_the_gap_to_the_row_minimum() -> None:
    import tests.test_posthoc_routing_20260928 as tr
    from vfe3.posthoc import routing as rt
    B, N, H, r = 2, 5, 3, 3
    m = tr._random_moments(B, N, H, r, seed=4)
    V_star = m["V_k"].mean(dim=(0, 1))
    W, V_inv, _ = rt.pooled_whitener(V_star)
    log_prior = torch.zeros(N, N, dtype=torch.float64).masked_fill(
        torch.ones(N, N, dtype=torch.bool).triu(1), float("-inf"))
    tau, kl_max = torch.full((H,), 1.5, dtype=torch.float64), 0.5
    comps = rt.logit_components(m["a_q"], m["V_q"], m["a_k"], m["V_k"], W, V_inv, log_prior, tau, kl_max,
                                kl_max_reference="row_min")
    D = comps["D"]
    row_min = D.masked_fill(~torch.isfinite(log_prior), math.inf).amin(-1, keepdim=True)
    torch.testing.assert_close(comps["reference"], row_min)
    gap = (D - row_min).clamp(0.0, kl_max)
    total = sum(comps[c] for c in rt.COMPONENTS)
    support = torch.isfinite(log_prior).expand_as(D)
    torch.testing.assert_close(total[support], (-(gap + row_min) / 1.5 + log_prior)[support])
    assert bool((comps["cap"][support & (D - row_min <= kl_max)] == 0).all())


def test_klcap_summary_records_the_reference() -> None:
    from vfe3.posthoc.klcap import KLCapObserver
    assert KLCapObserver(2.0, "row_min").summary()["kl_max_reference"] == "row_min"
    assert KLCapObserver(2.0).summary()["kl_max_reference"] == "absolute"


# --------------------------------------------------------------------------- item 23 / 44: viz

def test_readout_guard_fractions_read_the_block_spectrum() -> None:
    from vfe3.viz.extract import readout_state
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = test_config(vocab_size=9, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=1,
                          family="gaussian_block", decode_mode="full_chunked",
                          attention_mode="projected_predictive")
        torch.manual_seed(0)
        model = VFEModel(cfg).eval()
    tokens = torch.randint(0, 9, (1, 5), generator=torch.Generator().manual_seed(3))
    state = readout_state(model, tokens)
    assert state["sigma"].shape[-2:] == (4, 4)                               # dense K x K block covariance
    assert state["guard_denominators"]["sigma_floor_frac"] == state["sigma"].numel() // 4   # K eigenvalues per token


def test_readout_figure_draws_the_routing_cap_and_layer_max_columns(tmp_path) -> None:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    from vfe3.viz import figures as figs
    n = 12
    history = {"step": np.arange(1, n + 1) * 10.0, "attn_entropy": np.linspace(3, 2, n),
               "belief_cond_median": np.full(n, 1.6), "belief_cond_median_layer_max": np.full(n, 3.0),
               "relational_projected_cov_max_condition": np.full(n, 5.0),
               "relational_projected_cov_max_condition_layer_max": np.full(n, 9.0),
               "relational_score_klmax_frac": np.full(n, 0.1),
               "relational_score_klmax_frac_layer_max": np.full(n, 0.6),
               "relational_score_klmax_frac_argmax_layer": np.full(n, 1.0),
               "relational_score_klmax_frac_head_max": np.full(n, 0.7),
               "relational_clamped_row_frac_head_max": np.full(n, 0.0),
               "relational_self_weight_head_max": np.full(n, 0.99),
               "relational_self_argmax_frac_head_max": np.full(n, 1.0),
               "relational_self_lock_argmax_layer": np.full(n, 1.0)}
    fig = figs.plot_readout_diagnostics(history, path=str(tmp_path / "ro.png"))
    legends = {ax.get_title(): [t.get_text().split(" (last")[0] for t in ax.get_legend().get_texts()]
               for ax in fig.axes if ax.axison and ax.get_legend() is not None}
    assert "Routing cap" in legends and "Routing cap location" in legends
    assert "capped entries, last layer" in legends["Routing cap"]
    assert r"mean self weight $\beta_{ii}$, head max" in legends["Routing cap"]
    assert legends["Receiver covariance conditioning"] == ["median, last layer", "median, layer max"]
    assert "maximum, layer max" in legends["Comparison covariance conditioning"]
    figs.plt.close(fig)


def test_figure_parameter_groups_include_the_per_type_table() -> None:
    from vfe3.viz import figures as figs
    assert figs.PARAMETER_GROUPS == T.PARAMETER_GROUPS
    assert "per_type_table" in figs._TABLE_GROUPS
    assert all(group in figs._GROUP_LABELS and group in figs._GROUP_COLORS for group in figs.PARAMETER_GROUPS)


# --------------------------------------------------------------------------- item 33: block_cholesky readers

def test_posthoc_covariance_is_not_applicable_under_block_cholesky() -> None:
    from vfe3.posthoc.covariance import gap_applicability, hidden_temperature
    model = _cholesky_model()
    assert gap_applicability(model) == "the prior bank has no stored log-variance table"
    assert hidden_temperature(model)["status"] == "not_applicable"


def test_semantic_run_reads_the_block_cholesky_log_variance(tmp_path) -> None:
    import tests.test_semantics_core as sc
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from tests.profiles import tiny_config
        cfg = tiny_config(embed_dim=6, n_heads=2, vocab_size=sc.V, max_seq_len=5, n_layers=1,
                          family="gaussian_block", decode_mode="full_chunked", untie_decode_bank=True,
                          block_coordinates="reference", decode_coordinates="reference",
                          norm_type_block="layernorm", layernorm_affine=True, use_block_mlp=True,
                          block_mlp_mode="coordinate", content_link_chart="exponential", pos_phi="none",
                          gauge_transport="off", prior_covariance_table="block_cholesky")
        torch.manual_seed(0)
        model = VFEModel(cfg).eval()
    with torch.no_grad():
        model.prior_bank.sigma_chol_embed.add_(0.3 * torch.randn_like(model.prior_bank.sigma_chol_embed))
    run = sc._load(sc._save_run(tmp_path, "chol", cfg, model), cache_dir=tmp_path / "empty")
    ids = [1, 4, 7]
    with torch.no_grad():
        expected = model.prior_bank._prior_covariance_blocks(torch.tensor(ids)).diagonal(dim1=-2, dim2=-1)
    np.testing.assert_allclose(run.log_var(ids), expected.double().log().reshape(3, 6).numpy(), rtol=1e-6)
