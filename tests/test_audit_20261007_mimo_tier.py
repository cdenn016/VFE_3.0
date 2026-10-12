"""Tiny CPU regressions for the MiMo-tier items of the October 7, 2026 audit."""

import math
import types
import warnings

import pytest
import torch

from tests.profiles import test_config
from vfe3.config import ConfigNotice


def _cfg(**overrides):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConfigNotice)
        return test_config(**overrides)


# ---------------------------------------------------------------- item 6: routing support from the prior

def test_head_collapse_reads_support_from_the_prior_mask() -> None:
    from vfe3.metrics import attention_head_collapse

    beta = torch.zeros(1, 1, 4, 4)
    beta[..., torch.arange(4), torch.arange(4)] = 1.0          # one-hot rows: every other key underflowed
    support = torch.tril(torch.ones(4, 4, dtype=torch.bool))  # the prior admits the causal keys
    _worst, collapsed_inferred = attention_head_collapse(beta)
    worst, collapsed = attention_head_collapse(beta, support=support)
    assert not bool(collapsed_inferred.any())                  # beta > 0 sees one key per row: abstains
    assert bool(collapsed.all()) and float(worst.max()) < 1e-6


# ---------------------------------------------------------------- item 7: early minimal-agent refusal

def test_routing_arms_refuse_the_minimal_agent_before_any_pass() -> None:
    from vfe3.posthoc import interventions as iv

    model = types.SimpleNamespace(
        relational_operators_by_layer=(types.SimpleNamespace(predictive_readout="minimal_agent"),),
        cfg=types.SimpleNamespace(beta_attention_prior="causal"))
    for arm in (iv.routing_costs_removed, iv.prior_flattened, iv.routing_covariances_identity):
        with pytest.raises(iv.NotApplicable, match="minimal_agent"):
            with arm(model):
                raise AssertionError("the arm entered its body")


# ---------------------------------------------------------------- item 17: hidden temperature gate

def test_hidden_temperature_is_not_applicable_on_the_minimal_agent() -> None:
    from vfe3.posthoc.covariance import hidden_temperature

    model = types.SimpleNamespace(cfg=types.SimpleNamespace(predictive_readout="minimal_agent"))
    out = hidden_temperature(model)
    assert out["status"] == "not_applicable" and "channel noise" in out["reason"]


# ---------------------------------------------------------------- items 16, 31, 32: matched baseline

def test_matched_baseline_refuses_non_dense_bases() -> None:
    from vfe3.baseline import BaselineConfig

    for overrides, match in (({"predictive_readout": "minimal_agent"}, "predictive_readout"),
                             ({"use_block_mlp": False}, "no block MLP")):
        with pytest.raises(ValueError, match=match):
            BaselineConfig.matched_to(_cfg(attention_mode="projected_predictive", **overrides))
    with pytest.raises(ValueError, match="per_type_table"):
        BaselineConfig.matched_to(_cfg(attention_mode="projected_predictive", gauge_group="block_so",
                                       embed_dim=6, n_heads=2), per_type_table="block_read")


# ---------------------------------------------------------------- inactive_fields predicates

def test_closed_form_receiver_keeps_self_coupling_live_at_one_step() -> None:
    euler = _cfg(relational_receiver_update="euler").inactive_fields()
    closed = _cfg(relational_receiver_update="closed_form").inactive_fields()
    assert "lambda_alpha_mode" in euler
    assert "lambda_alpha_mode" not in closed


def test_unread_prior_and_cg_knobs_are_reported_inert() -> None:
    inactive = _cfg(beta_attention_prior="causal").inactive_fields()
    for name in ("alibi_slope", "t5_num_buckets", "t5_max_distance", "t5_learnable_bias",
                 "cg_covariance_mode", "cg_covariance_floor", "cg_energy_weight"):
        assert name in inactive, name
    assert "alibi_slope" not in _cfg(beta_attention_prior="causal_alibi").inactive_fields()


def test_closed_form_full_covariance_marks_sigma_max() -> None:
    closed_full = _cfg(relational_receiver_update="closed_form", family="gaussian_full",
                       skip_belief_sigma_update=False, lambda_alpha_mode="constant")
    assert "sigma_max" in closed_full.inactive_fields()


def test_lambda_alpha_is_validated_in_every_alpha_mode() -> None:
    with pytest.raises(ValueError, match="lambda_alpha"):
        _cfg(lambda_alpha_mode="state_dependent", lambda_alpha=-1.0)


def test_gauge_group_is_validated_against_the_registry() -> None:
    with pytest.raises(ValueError, match="gauge_group"):
        _cfg(gauge_group="not_a_group")


# ---------------------------------------------------------------- items 23, 57: pure-path report

def test_pure_path_report_labels_the_closed_form_update_and_reference_blocks() -> None:
    from vfe3.run_artifacts import _pure_path_report

    closed = _cfg(relational_receiver_update="closed_form", family="gaussian_full",
                  skip_belief_sigma_update=False, lambda_alpha_mode="constant")
    toggles = _pure_path_report(closed, [])["config_toggles"]
    assert toggles["spd_retraction_route"] == "closed_form_natural_parameter"
    assert toggles["spd_retraction_exact"] is True
    predictive = dict(attention_mode="projected_predictive", norm_type_block="layernorm")
    local = _pure_path_report(_cfg(**predictive), [])["gauge_flags"]
    reference = _pure_path_report(_cfg(**predictive, block_coordinates="reference"), [])["gauge_flags"]
    assert local["block_norm_gauge_equivariant"] is False
    assert reference["block_norm_gauge_equivariant"] is True


# ---------------------------------------------------------------- items 36, 37: host reads and cache

def test_frame_context_still_fails_closed_after_folding_its_reads() -> None:
    from vfe3.contracts import CanonicalFrameContext

    eye = torch.eye(3).expand(2, 3, 3).clone()
    CanonicalFrameContext(eye, eye.clone())
    with pytest.raises(ValueError, match="mutual inverses"):
        CanonicalFrameContext(eye, 2.0 * eye)
    with pytest.raises(ValueError, match="finite values"):
        CanonicalFrameContext(eye, torch.full_like(eye, math.nan))


def test_indexed_bounded_variance_equals_the_gathered_rows() -> None:
    from vfe3.numerics import bounded_variance_from_log

    table = torch.randn(5, 3)
    ids = torch.tensor([[0, 4, 4], [2, 1, 0]])
    assert torch.equal(bounded_variance_from_log(table, index=ids), bounded_variance_from_log(table[ids]))


# ---------------------------------------------------------------- items 41, 66: banner and BCH guard

def test_banner_tau_names_the_minimal_agent_temperature() -> None:
    from vfe3.train import _fmt_tau

    cfg = types.SimpleNamespace(attention_mode="projected_predictive", predictive_readout="minimal_agent")
    assert _fmt_tau(cfg, None).startswith("1 ")


def test_bch_composition_refuses_a_basis_that_is_not_bracket_closed() -> None:
    from vfe3.geometry.lie_ops import compose_bch

    generators = torch.zeros(2, 2, 2, dtype=torch.float64)
    generators[0, 0, 1] = 1.0                                   # E12
    generators[1, 1, 0] = 1.0                                   # E21; [E12, E21] = diag(1, -1) is outside
    phi = torch.tensor([0.1, 0.2], dtype=torch.float64)
    with pytest.raises(ValueError, match="not closed under the Lie bracket"):
        compose_bch(phi, phi, generators)


# ---------------------------------------------------------------- item 19: inert search parameters

def test_parameter_search_refuses_an_inert_enabled_parameter() -> None:
    import parameter_optimization as po

    specs = [("e_q_mu_lr", {"requires": {}})]
    with pytest.raises(ValueError, match="inert"):
        po._refuse_inert_parameters(specs, dict(vars(_cfg(attention_mode="projected_predictive"))), {})


# ---------------------------------------------------------------- reduced-precision replay ladder

def _ladder_fixture(monkeypatch: pytest.MonkeyPatch, behavior):
    import logging

    from vfe3 import train as train_module

    seen = []

    def fake_step(model, optimizer, scheduler, tokens, targets, **kwargs):
        policy = torch.get_float32_matmul_precision()
        seen.append((policy, float(torch.rand(()))))
        return behavior(policy, optimizer)

    monkeypatch.setattr(train_module, "train_step", fake_step)
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    model = types.SimpleNamespace(cfg=types.SimpleNamespace())

    def run(policy: str):
        previous = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision(policy)
        try:
            return train_module._train_step_with_precision_retry(
                model, optimizer, None, None, None, logger=logging.getLogger("t"), step=3, scaler=None)
        finally:
            assert torch.get_float32_matmul_precision() == policy        # restored after every rung
            torch.set_float32_matmul_precision(previous)
    return run, seen


def test_failed_step_escalates_medium_to_high_to_highest_with_the_same_rng(monkeypatch) -> None:
    def behavior(policy, optimizer):
        if policy != "highest":
            raise ValueError("canonical forward and inverse factors must be mutual inverses")
        return 1.5
    run, seen = _ladder_fixture(monkeypatch, behavior)
    torch.manual_seed(0)
    assert run("medium") == (1.5, 2)
    assert [policy for policy, _ in seen] == ["medium", "high", "highest"]
    assert len({draw for _, draw in seen}) == 1                          # every replay sees the same RNG


def test_failure_at_highest_and_failure_after_an_update_are_raised(monkeypatch) -> None:
    run, seen = _ladder_fixture(monkeypatch, lambda policy, optimizer: (_ for _ in ()).throw(ValueError("bug")))
    with pytest.raises(ValueError, match="bug") as info:
        run("high")
    assert [policy for policy, _ in seen] == ["high", "highest"]
    assert any("earlier attempt" in note for note in info.value.__notes__)

    def after_update(policy, optimizer):
        optimizer.step()
        raise ValueError("post-update failure")
    run, seen = _ladder_fixture(monkeypatch, after_update)
    with pytest.raises(ValueError, match="post-update"):
        run("high")
    assert len(seen) == 1                                                # never replayed after an update


def test_only_the_first_replay_warns(monkeypatch, caplog) -> None:
    import logging

    from vfe3 import train as train_module

    def flaky(policy, optimizer):
        if policy != "highest":
            raise ValueError("projected covariance is not positive definite")
        return 1.0

    _ladder_fixture(monkeypatch, flaky)
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    model = types.SimpleNamespace(cfg=types.SimpleNamespace())
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        with caplog.at_level(logging.DEBUG, logger="t"):
            for quiet in (False, True, True):
                assert train_module._train_step_with_precision_retry(
                    model, optimizer, None, None, None, logger=logging.getLogger("t"), step=3, quiet=quiet,
                    scaler=None) == (1.0, 1)
    finally:
        torch.set_float32_matmul_precision(previous)
    levels = [record.levelno for record in caplog.records if "replaying it at" in record.getMessage()]
    assert levels == [logging.WARNING, logging.DEBUG, logging.DEBUG]
    assert "summarized at each log step" in caplog.records[0].getMessage()


# ---------------------------------------------------------------- linear decode logits under autocast

def test_decode_logits_amp_runs_the_logits_matmul_in_bf16_and_returns_fp32() -> None:
    from vfe3.model.prior_bank import _linear_logits

    pb = types.SimpleNamespace(decode_logits_amp_dtype=torch.bfloat16)
    x, W = torch.randn(2, 3, 8), torch.randn(11, 8)
    logits = _linear_logits(pb, x, W)
    assert logits.dtype == torch.float32
    torch.testing.assert_close(logits, (x.bfloat16() @ W.bfloat16().T).float(), rtol=0, atol=0)
    assert torch.equal(_linear_logits(types.SimpleNamespace(decode_logits_amp_dtype=None), x, W), x @ W.T)


def test_decode_logits_amp_trains_a_tiny_linear_model_close_to_fp32() -> None:
    from vfe3.model.model import VFEModel

    losses = {}
    for flag in (False, True):
        cfg = _cfg(attention_mode="projected_predictive", decode_mode="linear", amp_dtype="bf16",
                   decode_logits_amp=flag)
        torch.manual_seed(0)
        model = VFEModel(cfg)
        ids = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
        out = model(ids, ids)
        loss = out[1] if isinstance(out, tuple) else out
        loss.backward()
        losses[flag] = float(loss)
        assert model.prior_bank.decode_logits_amp_dtype == (torch.bfloat16 if flag else None)
    assert math.isfinite(losses[True]) and abs(losses[True] - losses[False]) < 5e-2
    assert "decode_logits_amp" in _cfg(decode_mode="linear", amp_dtype=None).inactive_fields()
