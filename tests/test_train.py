from tests.profiles import config_values, test_config
import csv
import json
import logging
import math
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from vfe3.config import VFE3Config
from vfe3.data.datasets import TokenWindows
from vfe3.model.model import VFEModel
from vfe3.run_artifacts import RunArtifacts, finalize_validation_run
from vfe3.train import (
    TrainingTerminalState,
    build_optimizer,
    evaluate,
    lr_lambda,
    train,
    train_step,
    _floor_lr_lambdas,
)


def test_optimizer_groups_priors_by_m_lr():
    # A prior-bank decoder has no output_proj_weight group, so the four prior tables map to
    # exactly four LR groups (linear decode adds an output-projection group). The block MLP, the
    # untied decode tables and the affine block LayerNorm (all on by default) each add their own
    # group, so they are pinned off to isolate the prior-table grouping.
    cfg = test_config(vocab_size=20, embed_dim=4, n_heads=2,
                     m_p_mu_lr=0.01, m_p_sigma_lr=0.002, m_phi_lr=0.005,
                     decode_mode="diagonal_chunked",
                     use_block_mlp=False, untie_decode_bank=False, pos_phi="learned",
                     norm_type_block="none", layernorm_affine=False)
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)
    lrs = sorted(g["lr"] for g in opt.param_groups)
    # m_p_sigma_lr=0.002, then TWO groups at m_phi_lr=0.005 (phi_embed + the default pos_phi='learned'
    # pos_phi_free table, grouped at m_phi_lr in train.py), then m_p_mu_lr=0.01.
    assert lrs == [0.002, 0.005, 0.005, 0.01, 0.01, 0.01]
    # every PriorBank parameter is covered by exactly one group
    n_params = sum(len(g["params"]) for g in opt.param_groups)
    assert n_params == len([p for p in model.parameters() if p.requires_grad])


def test_parameter_report_leaves_global_rng_untouched():
    # parameter_report's probe forward must preserve the global RNG stream.
    from vfe3.train import parameter_report
    cfg = test_config(vocab_size=12, embed_dim=4, n_heads=2, max_seq_len=6, n_layers=1, n_e_steps=4)
    model = VFEModel(cfg)
    torch.manual_seed(123)
    before = torch.get_rng_state()
    parameter_report(model)
    assert torch.equal(before, torch.get_rng_state()), "parameter_report advanced the global RNG"


def test_lr_lambda_warmup_then_cosine():
    cfg = test_config(warmup_steps=10, max_steps=100)
    assert abs(lr_lambda(0, cfg) - 0.0) < 1e-6
    assert abs(lr_lambda(10, cfg) - 1.0) < 1e-6            # peak at end of warmup
    assert lr_lambda(55, cfg) < 1.0 and lr_lambda(55, cfg) > 0.0
    assert abs(lr_lambda(100, cfg) - 0.0) < 1e-3           # ~0 at max_steps


def test_scheduler_min_lr_frac_zero_is_pure_cosine():
    # min_lr_frac=0.0 is the theoretically pure path: the floor max(0, cosine)=cosine, so the LR
    # decays to exactly zero at max_steps, identical to the unfloored half-cosine.
    cfg = test_config(vocab_size=20, embed_dim=4, n_heads=2,
                     warmup_steps=2, max_steps=10, min_lr_frac=0.0)
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)
    base_lrs = [g["lr"] for g in opt.param_groups]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _floor_lr_lambdas(base_lrs, cfg))
    for _ in range(cfg.max_steps):
        sched.step()
    for lr in sched.get_last_lr():
        assert abs(lr) < 1e-9                              # pure cosine reaches zero


def test_fractional_floor_scales_each_group_to_min_lr_frac_times_base():
    # Option B: min_lr_frac floors EACH group's absolute LR at min_lr_frac * its own base LR,
    # so the m_mu:m_sigma:m_phi base ratios are preserved into the cosine tail.
    cfg = test_config(vocab_size=20, embed_dim=4, n_heads=2,
                     warmup_steps=2, max_steps=10, min_lr_frac=0.01,
                     m_p_mu_lr=0.02, m_p_sigma_lr=0.004, m_phi_lr=0.01)
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)
    base_lrs = [g["lr"] for g in opt.param_groups]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _floor_lr_lambdas(base_lrs, cfg))
    for _ in range(cfg.max_steps + 5):                     # into the clamped tail
        sched.step()
    for base, lr in zip(base_lrs, sched.get_last_lr()):
        assert math.isclose(lr, cfg.min_lr_frac * base, rel_tol=1e-9)


def test_floor_lambdas_handle_zero_base_lr_without_dividing():
    # A deliberately frozen channel (m_phi_lr=0) gives a group with base LR 0; the fractional
    # floor min_lr_frac * 0 keeps it frozen.
    cfg = test_config(vocab_size=20, embed_dim=4, n_heads=2,
                     warmup_steps=2, max_steps=10, min_lr_frac=0.01, m_phi_lr=0.0)
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)
    base_lrs = [g["lr"] for g in opt.param_groups]
    assert 0.0 in base_lrs                                 # the frozen phi group is present
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _floor_lr_lambdas(base_lrs, cfg))  # must not raise
    for _ in range(cfg.max_steps + 5):
        sched.step()
    for base, lr in zip(base_lrs, sched.get_last_lr()):
        if base == 0.0:
            assert lr == 0.0                               # frozen stays frozen
        else:
            assert lr >= cfg.min_lr_frac * base - 1e-12


def test_aux_role_learning_rate_reporting_handles_active_and_inactive_roles():
    from vfe3.train import _learning_rates_by_aux_role

    groups = [
        {"lr_aux_role": "layernorm"},
        {"role": "mu"},
    ]
    reported = _learning_rates_by_aux_role(groups, [0.0011, 0.0123])

    assert reported["layernorm"] == pytest.approx(0.0011)
    assert math.isnan(reported["block_mlp"])

    with pytest.raises(RuntimeError, match="duplicate optimizer learning-rate auxiliary role"):
        _learning_rates_by_aux_role(
            [{"lr_aux_role": "block_mlp"}, {"lr_aux_role": "block_mlp"}],
            [0.0022, 0.0033],
        )


@pytest.mark.parametrize("block_mlp_lr, expected_lr", [(None, 0.0123), (0.0456, 0.0456)])
def test_optimizer_covers_mlp_parameters_once_at_resolved_lr(block_mlp_lr, expected_lr):
    cfg = test_config(
        vocab_size=20,
        embed_dim=4,
        n_heads=2,
        max_seq_len=4,
        n_layers=2,
        n_e_steps=1,
        use_block_mlp=True,
        m_p_mu_lr=0.0123,
        m_block_mlp_lr=block_mlp_lr,
        weight_decay=0.0789,
        mu_weight_decay=0.0,
    )
    model = VFEModel(cfg)
    optimizer = build_optimizer(model, cfg)

    mlp_groups = [
        group for group in optimizer.param_groups
        if group.get("lr_aux_role") == "block_mlp"
    ]
    assert len(mlp_groups) == 1
    assert mlp_groups[0]["role"] == "mu"
    assert mlp_groups[0]["lr"] == pytest.approx(expected_lr)
    assert mlp_groups[0]["weight_decay"] == pytest.approx(cfg.weight_decay)
    mlp_parameter_ids = {id(parameter) for parameter in model.block_mlps.parameters()}
    assert {id(parameter) for parameter in mlp_groups[0]["params"]} == mlp_parameter_ids
    all_grouped_ids = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
        if id(parameter) in mlp_parameter_ids
    ]
    assert len(all_grouped_ids) == len(mlp_parameter_ids)
    assert len(set(all_grouped_ids)) == len(mlp_parameter_ids)


# The active alphabet of the period-3 stream is {0,1,2}; a structure-BLIND predictor
# (one that learns only the unigram frequencies of the active tokens) is pinned at the
# marginal entropy ln(3) ~ 1.0986. Beating that floor by a margin is the discriminating
# evidence that the model learned the period-3 NEXT-TOKEN structure, not just the marginal.


def _periodic_loader(V=6, period=3, n=600, seq_len=8, batch_size=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    base = torch.arange(period).repeat(n // period + 2)         # 0,1,2,0,1,2,...
    ds = TokenWindows(base[: n].to(torch.long), seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True,
                      generator=g)


def test_run_training_applies_cfg_seed_and_deterministic_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os
    import vfe3.train as train_module
    from vfe3.runtime import seed_everything

    cfg = test_config(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                     n_e_steps=1, batch_size=2, deterministic=False, seed=37)
    loader = _periodic_loader(V=cfg.vocab_size, n=96, seq_len=cfg.max_seq_len,
                              batch_size=cfg.batch_size, seed=cfg.seed)
    monkeypatch.setattr(train_module, "make_dataloader", lambda *args, **kwargs: loader)
    monkeypatch.setattr(train_module, "train", lambda *args, **kwargs: [])

    was_algorithms = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    was_cudnn_deterministic = torch.backends.cudnn.deterministic
    was_cudnn_benchmark = torch.backends.cudnn.benchmark
    had_cublas = "CUBLAS_WORKSPACE_CONFIG" in os.environ
    was_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    try:
        seed_everything(999, deterministic=True)
        model, _ = train_module.run_training(cfg, dataset="synthetic", n_steps=0)
        assert model.cfg is cfg
        assert torch.initial_seed() == cfg.seed
        assert torch.are_deterministic_algorithms_enabled() is False
        assert torch.backends.cudnn.deterministic is False
        assert torch.backends.cudnn.benchmark is True
    finally:
        torch.use_deterministic_algorithms(was_algorithms, warn_only=was_warn_only)
        torch.backends.cudnn.deterministic = was_cudnn_deterministic
        torch.backends.cudnn.benchmark = was_cudnn_benchmark
        if had_cublas:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = was_cublas
        else:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)




def _structured_cfg() -> VFE3Config:
    """Tiny projected-gradient profile for training-state and accumulation contracts."""
    return test_config(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                      n_e_steps=3, e_q_mu_lr=0.3,
                      m_p_mu_lr=0.05, m_p_sigma_lr=0.01, m_phi_lr=0.05,
                      warmup_steps=5, max_steps=200)




# Promoted from xfail to a hard gate (audit t3, 2026-07-07 re-validation): the per-head-(per-irrep-block)
# beta model learns the period-3 next-token map to CE ~= 0.002, clearing the ln(3) - 0.05 floor by +1.05
# nats -- byte-identical on CPU and on an RTX 5090 across the 3 fixed seeds, so no LR re-tuning is needed
# (the old ~0.047 thin margin was the retired SINGLE-beta model). The negative control
# test_random_stream_does_not_clear_cutover_anchor guards the other side. Threshold deliberately NOT
# massaged (audit honesty rule).


def test_training_smoke_on_real_wikitext2_if_present():
    # SOFTER SMOKE (NOT a structure-learning proof): on a real wikitext-2 slice this asserts
    # only that training is FINITE (no NaN) and the CE DECREASES by a margin. The ~2-nat drop
    # here (10.825 -> ~8.78 from a uniform-init ln(V)=10.825) is achievable by learning the
    # unigram token distribution alone: the same config on the SAME tokens randomly PERMUTED
    # (sequential structure destroyed, only the marginal histogram surviving) still drops well
    # past the 0.05 bar. So this guards numerics and end-to-end wiring on real vocabulary; it
    # does NOT certify next-token structure learning. The structured period-3 cutover (above),
    # which beats the active-alphabet marginal entropy, is the learnability gate.
    import pytest
    from vfe3.data.datasets import load_cached_tokens
    try:
        toks = load_cached_tokens("wikitext-2", "validation")
    except FileNotFoundError:
        pytest.skip("wikitext-2 cache absent")
    torch.manual_seed(0)
    cfg = test_config(vocab_size=50257, embed_dim=4, n_heads=2, max_seq_len=16, n_layers=1,
                     n_e_steps=1, e_q_mu_lr=0.3,
                     m_p_mu_lr=0.05, m_p_sigma_lr=0.01, m_phi_lr=0.0, warmup_steps=3, max_steps=30)
    model = VFEModel(cfg)
    ds = TokenWindows(toks[:4000], 16)
    loader = DataLoader(ds, batch_size=8, shuffle=True, drop_last=True)
    losses = train(model, loader, cfg, n_steps=30)
    assert all(map(lambda x: x == x, losses))                   # finite (no NaN)
    assert losses[-1] < losses[0] - 0.05                        # CE decreases (marginal learning suffices)


def test_evaluate_returns_finite_ppl_bpc_consistent_with_ce():
    torch.manual_seed(0)
    cfg = _structured_cfg()
    model = VFEModel(cfg)
    loader = _periodic_loader(seed=0)
    m = evaluate(model, loader, tokens_per_char=1.0)
    assert set(m.keys()) == {"ce", "ppl", "bits_per_token", "bpc",
                                  "expected_targets", "scored_targets", "excluded_targets",
                                  "overlap_context_targets", "padding_targets"}
    assert all(math.isfinite(v) for v in m.values())
    assert m["expected_targets"] == m["scored_targets"] + m["excluded_targets"]
    assert m["ppl"] == pytest.approx(math.exp(m["ce"]))
    assert m["bits_per_token"] == pytest.approx(m["ce"] / math.log(2.0))
    assert m["bpc"] == pytest.approx(m["ce"] / math.log(2.0))


class _ScriptedDecodeStatsModel(torch.nn.Module):
    """Minimal evaluation double whose decoder excludes a scripted number of targets."""

    def __init__(self, batches):
        super().__init__()
        self.prior_bank = torch.nn.Module()
        self.prior_bank.phi_embed = torch.nn.Parameter(torch.zeros(()))
        self._batches = list(batches)

    def forward(self, tokens, targets=None, *, return_decode_stats=False):
        assert return_decode_stats, "evaluation must request the decoder-owned denominator"
        batch = self._batches.pop(0)
        ce_value, scored_tokens = batch[:2]
        excluded_tokens = batch[2] if len(batch) == 3 else targets.numel() - scored_tokens
        ce = self.prior_bank.phi_embed * 0.0 + ce_value
        from vfe3.model.prior_bank import DecodeCEResult
        stats = DecodeCEResult(
            ce=ce.detach(),
            scored_tokens=torch.tensor(scored_tokens, dtype=torch.int64, device=ce.device),
            excluded_tokens=torch.tensor(excluded_tokens, dtype=torch.int64, device=ce.device),
        )
        return None, ce, stats


def test_evaluate_weights_by_decoder_scored_tokens_and_preserves_result_keys():
    model = _ScriptedDecodeStatsModel([(2.0, 1), (4.0, 3)])
    loader = [
        (torch.zeros(1, 4, dtype=torch.long), torch.zeros(1, 4, dtype=torch.long)),
        (torch.zeros(1, 4, dtype=torch.long), torch.zeros(1, 4, dtype=torch.long)),
    ]
    result = evaluate(model, loader)
    assert set(result) == {"ce", "ppl", "bits_per_token", "bpc",
                                  "expected_targets", "scored_targets", "excluded_targets",
                                  "overlap_context_targets", "padding_targets"}
    assert result["ce"] == pytest.approx(3.5)
    assert result["expected_targets"] == 8
    assert result["scored_targets"] == 4
    assert result["excluded_targets"] == 4


def test_evaluate_rejects_a_corpus_with_zero_decoder_scored_tokens():
    model = _ScriptedDecodeStatsModel([(0.0, 0)])
    loader = [(torch.zeros(1, 2, dtype=torch.long), torch.full((1, 2), -100))]
    with pytest.raises(ValueError, match="no scored target tokens"):
        evaluate(model, loader)


def test_silent_and_logging_paths_are_bitwise_identical(caplog):
    torch.manual_seed(0)
    cfg_a = _structured_cfg()
    model_a = VFEModel(cfg_a)
    loader_a = _periodic_loader(seed=0)
    losses_silent = train(model_a, loader_a, cfg_a, n_steps=20)

    torch.manual_seed(0)
    cfg_b = _structured_cfg()
    model_b = VFEModel(cfg_b)
    loader_b = _periodic_loader(seed=0)
    with caplog.at_level("INFO"):
        losses_logged = train(model_b, loader_b, cfg_b, n_steps=20, log_interval=1, eval_interval=0)

    assert losses_silent == losses_logged           # exact: logging must not perturb the hot path
    assert any("Step 1/20" in r.message for r in caplog.records)


class _CountingLoader:
    """Yields a fixed list of (tokens, targets) batches, counting how many are consumed.

    Used to prove periodic evaluation consumes the complete validation iterable.
    """

    def __init__(self, batches):
        self.batches = batches
        self.count = 0

    def __iter__(self):
        for b in self.batches:
            self.count += 1
            yield b


def test_evaluate_max_batches_stops_after_exact_requested_count():
    model = _ScriptedDecodeStatsModel([
        (1.0, 2),
        (3.0, 2),
        (99.0, 2),
        (99.0, 2),
    ])
    batch = (
        torch.zeros(1, 4, dtype=torch.long),
        torch.zeros(1, 4, dtype=torch.long),
    )
    loader = _CountingLoader([batch, batch, batch, batch])

    result = evaluate(model, loader, max_batches=2)

    assert loader.count == 2
    assert result["ce"] == pytest.approx(2.0)
    assert result["expected_targets"] == 8
    assert result["scored_targets"] == 4
    assert len(model._batches) == 2


def test_train_periodic_eval_consumes_full_validation_loader():
    # With a five-batch validation loader and two periodic evaluations, every batch is scored.
    cfg = test_config(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                     n_e_steps=1, m_phi_lr=0.0, warmup_steps=1, max_steps=2)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    train_loader = _periodic_loader(V=6, period=3, seq_len=8, batch_size=4, seed=0)
    val_batches = [(torch.randint(0, 3, (4, 8)), torch.randint(0, 3, (4, 8))) for _ in range(5)]
    val_loader = _CountingLoader(val_batches)
    train(model, train_loader, cfg, n_steps=2, eval_interval=1, val_loader=val_loader)
    assert val_loader.count == 10


def test_sample_decode_emits_sample_line_each_eval(caplog):
    # When a decoder is supplied, train() prints a "Sample:" line below the BPC value every eval;
    # when it is None (the default) nothing is generated. A decode/gen error must not raise.
    cfg = test_config(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                     n_e_steps=1, m_phi_lr=0.0, warmup_steps=1, max_steps=2)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    decode = lambda ids: " ".join(str(int(t)) for t in ids)        # trivial token->text decoder
    with caplog.at_level("INFO"):
        train(model, _periodic_loader(seed=0), cfg, n_steps=2, eval_interval=1,
              val_loader=_periodic_loader(seed=1), sample_decode=decode,
              sample_new_tokens=3, sample_prompt_len=4)
    assert sum("Sample:" in r.message for r in caplog.records) == 2  # one per eval step


def test_tiny_vocab_auto_default_stays_silent(caplog):
    # The vocab-gated auto-default decoder is None for a tiny synthetic/test vocab (6 is not a
    # real tokenizer size), so with no explicit decoder no Sample line is emitted -- the pure
    # silent path is preserved with no extra toggle.
    cfg = test_config(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                     n_e_steps=1, m_phi_lr=0.0, warmup_steps=1, max_steps=2)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    with caplog.at_level("INFO"):
        train(model, _periodic_loader(seed=0), cfg, n_steps=2, eval_interval=1,
              val_loader=_periodic_loader(seed=1))
    assert not any("Sample:" in r.message for r in caplog.records)


def test_generate_samples_false_is_silent_at_real_vocab(caplog):
    # The pure-path opt-out: generate_samples=False suppresses sampling even at a real gpt2 vocab
    # where the auto-default would otherwise fire (CLAUDE.md: a reachable silent path under a toggle).
    pytest.importorskip("tiktoken")
    cfg = test_config(vocab_size=50257, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                     n_e_steps=1, m_phi_lr=0.0, warmup_steps=1, max_steps=1)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    with caplog.at_level("INFO"):
        train(model, _periodic_loader(seed=0), cfg, n_steps=1, eval_interval=1,
              val_loader=_periodic_loader(seed=1), generate_samples=False)
    assert not any("Sample:" in r.message for r in caplog.records)


def test_auto_default_sample_decoder_emits_at_gpt2_vocab(caplog):
    # At a real gpt2 vocab the auto-default decoder activates with NO explicit decoder and NO
    # entry-file wiring: a Sample line prints below BPC each eval (skip if tiktoken is absent).
    pytest.importorskip("tiktoken")
    cfg = test_config(vocab_size=50257, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                     n_e_steps=1, m_phi_lr=0.0, warmup_steps=1, max_steps=1)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    with caplog.at_level("INFO"):
        train(model, _periodic_loader(seed=0), cfg, n_steps=1, eval_interval=1,
              val_loader=_periodic_loader(seed=1), sample_new_tokens=3, sample_prompt_len=4)
    assert any("Sample:" in r.message for r in caplog.records)


def test_train_vfe3_imports_and_a_local_operating_point_runs_one_step(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""The launcher module imports, and its config construction runs one step on a local test operating point that
    replaces ``train_vfe3.config``; the launcher dictionary itself is not read here (the live route is covered by
    tests/test_live_route_profile.py; audit 2026-10-17, item 42)."""
    import train_vfe3

    # Exercise the launcher's config construction with a complete local operating point.
    local = config_values(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=4,
                          batch_size=2, n_layers=1, n_e_steps=1, max_steps=2,
                          warmup_steps=1, eval_stride=None)
    monkeypatch.setattr(train_vfe3, "config", local)
    cfg = train_vfe3.VFE3Config(**train_vfe3.config)
    loader = _periodic_loader(n=cfg.max_seq_len * cfg.batch_size * 4,
                              seq_len=cfg.max_seq_len, batch_size=cfg.batch_size, seed=cfg.seed)
    batch = next(iter(loader))
    assert len(batch) == 2
    torch.manual_seed(cfg.seed)
    model = VFEModel(cfg)
    losses = train(model, loader, cfg, n_steps=1)
    assert len(losses) == 1 and math.isfinite(losses[0])


def test_build_optimizer_groups_pos_phi_free():
    cfg = test_config(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                     pos_phi="learned", m_phi_lr=0.009)
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)                      # must NOT raise the coverage AssertionError
    grouped = {p for g in opt.param_groups for p in g["params"]}
    assert model.pos_phi_free in grouped


def test_train_step_skips_on_nonfinite_grad_with_finite_loss():
    r"""F1 (audit 2026-07-01): a FINITE scalar loss can still carry a NaN parameter gradient
    through the unrolled E-step; the finite-GRADIENT gate must skip the optimizer step so AdamW's
    exp_avg/exp_avg_sq moment buffers are never poisoned. The scheduler is an accepted-update
    transition and must remain unchanged when that update is rejected."""
    torch.manual_seed(0)
    cfg = test_config(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                     n_e_steps=1, e_q_mu_lr=0.1, m_phi_lr=0.0,
                     warmup_steps=1, max_steps=4)
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, cfg))
    g = torch.Generator().manual_seed(0)
    tokens = torch.randint(0, 6, (2, 8), generator=g)
    targets = torch.randint(0, 6, (2, 8), generator=g)

    train_step(model, opt, sched, tokens, targets, grad_clip=1.0)   # clean step populates AdamW state
    assert opt.state                                                # moment buffers exist (non-vacuous)
    # A hook that poisons ONE parameter's gradient while the forward (and its scalar loss) stays finite.
    model.prior_bank.mu_embed.register_hook(lambda grad: torch.full_like(grad, float("nan")))
    metrics = {}
    loss = train_step(model, opt, sched, tokens, targets, grad_clip=1.0, metrics_out=metrics)

    assert math.isfinite(loss)
    assert metrics["loss_finite"] == 1.0                            # the scalar loss WAS finite
    assert metrics["grad_finite"] == 0.0                            # ...but the gradient was not
    assert metrics["step_skipped"] == 1.0                           # so the optimizer step was skipped
    assert all(torch.isfinite(p).all() for p in model.parameters())
    for state in opt.state.values():                                # AdamW moments stay clean
        for key in ("exp_avg", "exp_avg_sq"):
            if key in state:
                assert torch.isfinite(state[key]).all()
    assert sched.last_epoch == 1                                    # rejected update leaves schedule unchanged


# =============================================================================
# Gradient-clipping runtime pinning (PB-15): None/zero disable,
# fixed per-role clip counts, omitted-vs-explicit equivalence, terminal-callback seam.
# =============================================================================

def _clip_case():
    cfg = test_config(
        vocab_size=8, embed_dim=4, n_heads=2, max_seq_len=4,
    )
    model = VFEModel(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_lambda(step, cfg)
    )
    tokens = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    targets = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    return model, optimizer, scheduler, tokens, targets


def _install_clip_spy(monkeypatch):
    calls = []

    def spy(parameters, max_norm, *args, **kwargs):
        params = list(parameters)
        calls.append(({id(p) for p in params}, float(max_norm)))

    # train_step clips by get_total_norm (before its fused host transfer) + clip_grads_with_norm_.
    monkeypatch.setattr(torch.nn.utils, "clip_grads_with_norm_", spy)
    return calls


def test_grad_clip_roles_do_not_rescale_each_other(monkeypatch):
    model, optimizer, scheduler, tokens, targets = _clip_case()
    parameters = [group["params"][0] for group in optimizer.param_groups[:2]]
    assert optimizer.param_groups[0]["role"] != optimizer.param_groups[1]["role"]
    calls = _install_clip_spy(monkeypatch)
    train_step(model, optimizer, scheduler, tokens, targets, grad_clip=0.25)
    assert not any(all(id(parameter) in ids for parameter in parameters) for ids, _ in calls)


def test_grad_clip_per_role_calls_once_per_nonempty_role(monkeypatch):
    model, optimizer, scheduler, tokens, targets = _clip_case()
    expected = {}
    for group in optimizer.param_groups:
        expected.setdefault(group.get("role", "other"), set()).update(
            id(p) for p in group["params"]
        )
    calls = _install_clip_spy(monkeypatch)
    # A threshold below every role norm binds every role: a role whose coefficient is exactly 1 skips the
    # multiply (bit-identical; audit 2026-10-17 item 70), so a binding threshold is needed to see every call.
    train_step(model, optimizer, scheduler, tokens, targets, grad_clip=1e-9)
    assert {frozenset(ids) for ids, _ in calls} == {
        frozenset(ids) for ids in expected.values() if ids
    }
    assert all(max_norm == 1e-9 for _, max_norm in calls)
    flattened = [pid for ids, _ in calls for pid in ids]
    assert len(flattened) == len(set(flattened))


@pytest.mark.parametrize("grad_clip", [None, 0.0])
def test_grad_clip_off_still_steps(monkeypatch, grad_clip):
    model, optimizer, scheduler, tokens, targets = _clip_case()
    calls = _install_clip_spy(monkeypatch)
    status = {}
    before_epoch = scheduler.last_epoch
    train_step(
        model, optimizer, scheduler, tokens, targets,
        grad_clip=grad_clip, status_out=status,
    )
    assert calls == []
    assert status["did_step"] is True
    assert scheduler.last_epoch == before_epoch + 1


def test_omitted_grad_clip_matches_explicit_one(monkeypatch):
    left, left_opt, left_sched, tokens, targets = _clip_case()
    right, right_opt, right_sched, _, _ = _clip_case()
    right.load_state_dict(left.state_dict())
    right_opt.load_state_dict(left_opt.state_dict())
    right_sched.load_state_dict(left_sched.state_dict())
    original = torch.nn.utils.clip_grads_with_norm_
    calls = []

    def tracked(parameters, max_norm, *args, **kwargs):
        params = list(parameters)
        calls.append((len(params), float(max_norm)))
        return original(params, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grads_with_norm_", tracked)
    train_step(left, left_opt, left_sched, tokens, targets)
    split = len(calls)
    train_step(right, right_opt, right_sched, tokens, targets, grad_clip=1.0)
    assert calls[:split] == calls[split:]
    for key, value in left.state_dict().items():
        assert torch.equal(value, right.state_dict()[key])


def test_grad_clip_signature_preserves_terminal_callback():
    r"""PB-15 seam guard: the clipping annotation/wiring edit must not erase the PB-02
    terminal-callback contract -- train() still invokes it exactly once with a
    TrainingTerminalState, even when grad_clip is threaded through explicitly."""
    cfg = test_config(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                     n_e_steps=1, m_phi_lr=0.0, warmup_steps=1, max_steps=1)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    captured = {"calls": 0}

    def cb(state, callback_losses):
        captured["calls"] += 1
        captured["state"] = state

    train(model, _periodic_loader(seed=0), cfg, n_steps=1,
          terminal_callback=cb, grad_clip=cfg.grad_clip)

    assert captured["calls"] == 1
    assert isinstance(captured["state"], TrainingTerminalState)


def test_attention_diagnostic_failure_does_not_kill_training(tmp_path, monkeypatch, caplog):
    r"""F11 (audit 2026-07-01): a held-out attention diagnostic failure must be caught by the
    caller-side guard (warn + continue), never abort training."""
    cfg = test_config(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                     n_e_steps=1, m_phi_lr=0.0, warmup_steps=1, max_steps=2)
    torch.manual_seed(0)
    model = VFEModel(cfg)

    def _boom(*a, **k):
        raise RuntimeError("replay boom")

    monkeypatch.setattr(model, "build_diagnostic_snapshot", _boom)
    art = RunArtifacts(tmp_path / "run", cfg, model)
    with caplog.at_level(logging.WARNING):
        losses = train(model, _periodic_loader(seed=0), cfg, n_steps=2, eval_interval=1,
                       val_loader=_periodic_loader(seed=1), artifacts=art)
    assert len(losses) == 2                                     # train() completed despite the failure
    assert all(math.isfinite(loss) for loss in losses)
    assert art.diagnostics_failures["training_diagnostics"] == 2
    assert any("validation diagnostics failed" in r.getMessage() for r in caplog.records)
    with (tmp_path / "run" / "metrics.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert all(row["attn_entropy"] == "" for row in rows)
    assert all(int(row["diagnostics_failures"]) >= 1 for row in rows)


def test_val_diagnostics_failure_resets_to_nan(tmp_path, monkeypatch):
    r"""F11 (audit 2026-07-01): a _val_diagnostics failure must RESET the held-out probes to NaN
    (blank CSV cells) instead of carrying the previous eval's values forward as if fresh."""
    import vfe3.train as vt

    cfg = test_config(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                     n_e_steps=1, m_phi_lr=0.0, warmup_steps=1, max_steps=2)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    calls = {"n": 0}
    original_val_diagnostics = vt._val_diagnostics

    def _flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            current = original_val_diagnostics(*a, **k)
            metrics = dict(current.metrics)
            metrics.update({"val_inner_alignment_energy_total": 1.23})
            return vt.ValidationDiagnostics(
                metrics,
                current.token_ids,
                current.snapshot,
            )                                                    # eval 1: a fresh probe value
        raise RuntimeError("diagnostics boom")                  # eval 2: replay failure

    monkeypatch.setattr(vt, "_val_diagnostics", _flaky)
    art = RunArtifacts(tmp_path / "run", cfg, model)
    train(model, _periodic_loader(seed=0), cfg, n_steps=2, eval_interval=1,
          val_loader=_periodic_loader(seed=1), artifacts=art)
    with open(tmp_path / "run" / "metrics.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert calls["n"] == 2                                      # both evals hit the probe
    assert rows[0]["val_inner_alignment_energy_total"] == "1.23"           # eval 1 wrote the fresh value
    assert rows[1]["val_inner_alignment_energy_total"] == ""               # eval 2 failed -> NaN -> blank, NOT stale 1.23


def test_select_loader_is_split_aware(monkeypatch):
    r"""Audit F1: _select_loader must request shuffle=False, drop_last=False for validation/test
    (stable corpus metric) and shuffle=True, drop_last=True only for train. RED against the old
    _select_loader, which called make_dataloader with neither override (so val/test inherited the
    shuffle=True, drop_last=True training defaults)."""
    import train_vfe3

    captured = {}

    def fake_make_dataloader(dataset, split, seq_len, batch_size, *,
                             shuffle=True, drop_last=True, max_tokens=None, **kw):
        captured[split] = {"shuffle": shuffle, "drop_last": drop_last}
        return ("loader", split)

    monkeypatch.setattr(train_vfe3, "make_dataloader", fake_make_dataloader)
    cfg = test_config()
    for split in ("train", "validation", "test"):
        train_vfe3._select_loader("wikitext-103", cfg, split=split)

    assert captured["train"] == {"shuffle": True, "drop_last": True}
    assert captured["validation"] == {"shuffle": False, "drop_last": False}
    assert captured["test"] == {"shuffle": False, "drop_last": False}


def test_select_loader_raises_on_missing_cache(monkeypatch):
    r"""H1: a missing real-corpus cache must RAISE, never silently substitute the synthetic
    period-3 stream. The old fallback computed held-out numbers on a 3-token toy stream and
    persisted them mislabeled as the real corpus (config.json / test_results.json / run-folder name)."""
    import train_vfe3

    def raise_fnf(*a, **kw):
        raise FileNotFoundError("no cache")

    monkeypatch.setattr(train_vfe3, "make_dataloader", raise_fnf)
    cfg = test_config()
    for split in ("train", "validation", "test"):
        with pytest.raises(FileNotFoundError):
            train_vfe3._select_loader("wikitext-103", cfg, split=split)


# =============================================================================
# Terminal callback + validation-only finalizer (PB-02, audit 2026-07-12)
# =============================================================================
# A default ablation cell (log/eval interval above max_steps, checkpoint_interval=0) trains fine but
# used to save no metrics.csv / best_model.pt / resumable bundle. train() now invokes a terminal
# callback once, immediately after the final optimizer step, and finalize_validation_run() turns that
# snapshot into a complete VALIDATION-ONLY artifact set (never a test split) plus a resumable checkpoint
# whose weights and optimizer moments describe the same final iterate.
# Tiny CPU models throughout: embed_dim=4, n_heads=2, n_layers=1, n_e_steps=1, two batches.


def _terminal_cfg(**kw):
    base = dict(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                n_e_steps=1, gauge_group="block_glk",
                e_q_mu_lr=0.5, e_q_sigma_lr=0.05,
                m_p_mu_lr=0.1, m_p_sigma_lr=0.05, m_phi_lr=0.0,
                warmup_steps=1, max_steps=2, kl_max=32)
    base.update(kw)
    return test_config(**base)


def _terminal_loader(seed=7, n=64, seq_len=8, bs=4):
    # Two batches (8 windows / bs 4, drop_last); shuffled with an explicit generator so the terminal
    # checkpoint captures a resumable data cursor.
    g = torch.Generator().manual_seed(seed)
    base = torch.arange(3).repeat(n // 3 + 2)
    ds = TokenWindows(base[:n].long(), seq_len)
    return DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True, generator=g)


def _terminal_val_loader(n=48, seq_len=8, bs=4):
    base = torch.arange(3).repeat(n // 3 + 2)
    ds = TokenWindows(base[:n].long(), seq_len)
    return DataLoader(ds, batch_size=bs, shuffle=False, drop_last=False)


def _make_terminal_state(model, cfg):
    r"""A real TrainingTerminalState built from the model's current weights, optimizer, and RNG."""
    opt = build_optimizer(model, cfg)
    raw = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    rng = {"cpu": torch.get_rng_state().clone(),
           "cuda": ([s.clone() for s in torch.cuda.get_rng_state_all()]
                    if torch.cuda.is_available() else None)}
    return TrainingTerminalState(
        step=int(cfg.max_steps), optimizer=opt, scaler=None,
        data_state=None, raw_model_state=raw, rng_state=rng)


def test_train_terminal_callback_receives_resumable_state():
    torch.manual_seed(0)
    cfg = _terminal_cfg()
    model = VFEModel(cfg)
    loader = _terminal_loader()
    captured = {"calls": 0}

    def cb(state, callback_losses):
        captured["calls"] += 1
        captured["state"] = state
        captured["losses_len"] = len(callback_losses)

    train(model, loader, cfg, n_steps=cfg.max_steps, terminal_callback=cb)

    assert captured["calls"] == 1                                   # invoked EXACTLY once
    st = captured["state"]
    assert st.step == cfg.max_steps                                 # the completed-step count
    assert isinstance(st.optimizer, torch.optim.Optimizer)
    assert set(st.raw_model_state) == set(model.state_dict())       # full state_dict, strictly reloadable
    assert "cpu" in st.rng_state and st.rng_state["cpu"] is not None
    assert captured["losses_len"] == cfg.max_steps
    # With no post-training weight transform, the callback snapshot is the returned final iterate.
    final = model.state_dict()
    assert all(torch.equal(st.raw_model_state[n], final[n]) for n in final)


def test_validation_finalizer_records_validation_without_test_fields(tmp_path, monkeypatch):
    monkeypatch.setattr("vfe3.run_artifacts._git_code_identity",
                        lambda *a, **k: {"git_sha": "0" * 40, "git_dirty": False,
                                         "git_dirty_fingerprint": None})
    torch.manual_seed(0)
    cfg = _terminal_cfg()
    model = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "r", cfg, model)
    state = _make_terminal_state(model, cfg)
    mapping = finalize_validation_run(
        model, art, cfg, _terminal_val_loader(), losses=[1.0, 0.9],
        terminal_state=state, device=torch.device("cpu"))

    summary = json.loads((tmp_path / "r" / "summary.json").read_text())
    assert summary["selection_split"] == "validation"
    for forbidden in ("test_ce", "test_ppl", "test_bpc"):
        assert forbidden not in summary
    for required in ("primary_val_ppl", "final_val_ce", "final_val_ppl",
                     "final_val_bits_per_token", "final_val_bpc",
                     "expected_targets", "scored_targets", "excluded_targets",
                     "val_expected_targets", "val_scored_targets", "val_excluded_targets",
                     "best_val_ppl", "best_step", "n_steps", "n_params", "final_train_loss",
                     "wall_time_s", "terminal_checkpoint", "figures_written",
                     "phi_chart_norm_route"):
        assert required in summary
    assert isinstance(summary["figures_written"], list)
    assert summary["phi_chart_norm_route"] is None

    vr = json.loads((tmp_path / "r" / "validation_results.json").read_text())
    assert vr["selection_split"] == "validation"
    assert "test_ppl" not in vr

    assert set(mapping) == {"primary_val_ppl", "final_val_ppl", "final_val_ce",
                            "final_val_bits_per_token", "final_val_bpc",
                            "expected_targets", "scored_targets", "excluded_targets",
                            "val_expected_targets", "val_scored_targets", "val_excluded_targets",
                            "best_val_ppl", "best_step", "final_train_loss", "n_params",
                            "terminal_checkpoint", "save_model_artifacts",
                            "skipped_steps", "accepted_update_frac",
                            "terminal_selection_published", "unpublished_terminal_val_ppl"}
    # After the terminal maybe_save_best the primary equals the selected finite best (no earlier best).
    assert mapping["primary_val_ppl"] == mapping["best_val_ppl"] == mapping["final_val_ppl"]
    assert Path(mapping["terminal_checkpoint"]).exists()
    assert (tmp_path / "r" / "checkpoints" / f"step_{cfg.max_steps}.pt").exists()


def test_validation_finalizer_uses_completed_terminal_step_not_configured_max(tmp_path, monkeypatch):
    monkeypatch.setattr("vfe3.run_artifacts._git_code_identity",
                        lambda *a, **k: {"git_sha": "0" * 40, "git_dirty": False,
                                         "git_dirty_fingerprint": None})
    cfg = _terminal_cfg(max_steps=4)
    model = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "r", cfg, model)
    state = _make_terminal_state(model, cfg)._replace(step=2)

    mapping = finalize_validation_run(
        model, art, cfg, _terminal_val_loader(), losses=[1.0, 0.9],
        terminal_state=state, device=torch.device("cpu"))

    summary = json.loads((tmp_path / "r" / "summary.json").read_text())
    assert summary["n_steps"] == 2
    assert summary["best_step"] == 2
    assert Path(mapping["terminal_checkpoint"]).name == "step_2.pt"
    assert not (tmp_path / "r" / "checkpoints" / "step_4.pt").exists()
    with open(tmp_path / "r" / "metrics.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[-1]["step"] == "2"


def test_validation_finalizer_appends_to_existing_metrics_schema(tmp_path, monkeypatch):
    monkeypatch.setattr("vfe3.run_artifacts._git_code_identity",
                        lambda *a, **k: {"git_sha": "0" * 40, "git_dirty": False,
                                         "git_dirty_fingerprint": None})
    torch.manual_seed(0)
    cfg = _terminal_cfg()
    model = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "r", cfg, model)
    # An established (richer) training schema locks the CSV fieldnames on its first row; the terminal
    # row's metric keys are a subset, so the append writes blanks for other columns.
    art.log_metrics({"step": 1, "train_loss": 1.0, "train_ce": 1.0, "val_ce": float("nan"),
                     "val_ppl": float("nan"), "val_bits_per_token": float("nan"),
                     "val_bpc": float("nan"), "lr_mu": 0.01,
                     "attn_entropy": 0.5, "generalization_gap": 0.25})
    state = _make_terminal_state(model, cfg)
    finalize_validation_run(model, art, cfg, _terminal_val_loader(), losses=[1.0, 0.9],
                            terminal_state=state, device=torch.device("cpu"))

    with open(tmp_path / "r" / "metrics.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert "attn_entropy" in rows[0]                                # original schema preserved
    assert rows[-1]["step"] == str(cfg.max_steps)                   # terminal row appended
    assert rows[-1]["val_ppl"] != ""                               # terminal validation recorded
    for stale_field in (
        "train_loss", "train_ce", "lr_mu", "attn_entropy", "generalization_gap",
    ):
        assert rows[-1][stale_field] == ""                          # non-terminal columns left blank


def test_terminal_callback_restores_cpu_and_cuda_rng(tmp_path, monkeypatch):
    monkeypatch.setattr("vfe3.run_artifacts._git_code_identity",
                        lambda *a, **k: {"git_sha": "0" * 40, "git_dirty": False,
                                         "git_dirty_fingerprint": None})
    torch.manual_seed(0)
    cfg = _terminal_cfg()
    model = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "r", cfg, model)
    val = _terminal_val_loader()
    captured = {}

    def cb(state, callback_losses):
        # Global RNG captured immediately after the last training step (== state.rng_state).
        captured["cpu"] = state.rng_state["cpu"].clone()
        captured["cuda"] = ([s.clone() for s in state.rng_state["cuda"]]
                            if state.rng_state["cuda"] is not None else [])
        # The finalizer draws RNG (eval + figures) then restores the captured state in its finally.
        finalize_validation_run(model, art, cfg, val, losses=callback_losses,
                                terminal_state=state, device=torch.device("cpu"))

    train(model, _terminal_loader(), cfg, n_steps=cfg.max_steps, artifacts=art,
          val_loader=val, terminal_callback=cb)

    assert torch.equal(torch.get_rng_state(), captured["cpu"])      # CPU stream rewound
    cuda_now = ([s for s in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available() else [])               # empty list on a CPU-only host
    assert [s.tolist() for s in cuda_now] == [s.tolist() for s in captured["cuda"]]


def test_terminal_checkpoint_resumes_optimizer_rng_and_next_step(tmp_path, monkeypatch):
    monkeypatch.setattr("vfe3.run_artifacts._git_code_identity",
                        lambda *a, **k: {"git_sha": "0" * 40, "git_dirty": False,
                                         "git_dirty_fingerprint": None})
    torch.manual_seed(0)
    cfg = _terminal_cfg(max_steps=2)
    model = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "r", cfg, model)
    val = _terminal_val_loader()
    loader = _terminal_loader()
    result = {}

    def cb(state, callback_losses):
        result.update(finalize_validation_run(
            model, art, cfg, val, losses=callback_losses, train_loader=loader,
            terminal_state=state, device=torch.device("cpu")))

    train(model, loader, cfg, n_steps=cfg.max_steps, artifacts=art,
          val_loader=val, terminal_callback=cb)

    ckpt = result["terminal_checkpoint"]
    assert Path(ckpt).exists()
    bundle = torch.load(ckpt, weights_only=True)                   # safe-loads (tensors + config only)
    assert bundle["step"] == cfg.max_steps
    for slot in ("model_state", "optimizer_state", "rng_state", "scaler_state",
                 "data_state", "best_val_ppl", "best_step"):
        assert slot in bundle
    assert "ema_state" not in bundle

    # Resume: a fresh model + optimizer continues from the saved step for EXACTLY one more step.
    torch.manual_seed(0)
    cfg_resume = _terminal_cfg(max_steps=cfg.max_steps + 1)
    fresh = VFEModel(cfg_resume)
    losses_resume = train(fresh, _terminal_loader(), cfg_resume, n_steps=cfg.max_steps + 1,
                          resume_from=ckpt, device=torch.device("cpu"))
    assert len(losses_resume) == 1                                 # start_step == max_steps -> one step
    assert math.isfinite(losses_resume[0])


def test_terminal_checkpoint_weights_resume_exactly(tmp_path, monkeypatch):
    monkeypatch.setattr("vfe3.run_artifacts._git_code_identity",
                        lambda *a, **k: {"git_sha": "0" * 40, "git_dirty": False,
                                         "git_dirty_fingerprint": None})
    # Uninterrupted three-step control.
    torch.manual_seed(0)
    cfg_ctrl = _terminal_cfg(max_steps=3)
    ctrl = VFEModel(cfg_ctrl)
    train(ctrl, _terminal_loader(), cfg_ctrl, n_steps=3, device=torch.device("cpu"))
    ctrl_raw = {n: p.detach().clone() for n, p in ctrl.named_parameters() if p.requires_grad}

    # Interrupted two-step first leg with terminal validation/checkpoint publication.
    torch.manual_seed(0)
    cfg_first = _terminal_cfg(max_steps=2)
    interrupted = VFEModel(cfg_first)
    art = RunArtifacts(tmp_path / "r", cfg_first, interrupted)
    val = _terminal_val_loader()
    loader = _terminal_loader()
    grab = {}

    def cb(state, callback_losses):
        grab["raw2"] = {k: v.clone() for k, v in state.raw_model_state.items()}
        grab.update(finalize_validation_run(
            interrupted, art, cfg_first, val, losses=callback_losses, train_loader=loader,
            terminal_state=state, device=torch.device("cpu")))

    train(interrupted, loader, cfg_first, n_steps=2, artifacts=art,
          val_loader=val, terminal_callback=cb)
    ckpt = grab["terminal_checkpoint"]

    final_first = {n: p.detach().clone() for n, p in interrupted.named_parameters()
                   if p.requires_grad}
    assert all(torch.equal(final_first[n], grab["raw2"][n]) for n in final_first)
    best_bundle = torch.load(tmp_path / "r" / "best_model.pt", weights_only=True)
    assert all(torch.equal(best_bundle["model_state"][n], grab["raw2"][n]) for n in final_first)

    # Resume the saved iterate and optimizer moments for one step and match the control.
    torch.manual_seed(0)
    cfg_res = _terminal_cfg(max_steps=3)
    resumed = VFEModel(cfg_res)
    train(resumed, _terminal_loader(), cfg_res, n_steps=3, resume_from=ckpt, device=torch.device("cpu"))
    for n, p in resumed.named_parameters():
        if p.requires_grad:
            assert torch.allclose(p.detach(), ctrl_raw[n], atol=1e-5, rtol=1e-4), n
