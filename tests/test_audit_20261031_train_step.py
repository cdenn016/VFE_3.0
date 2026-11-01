r"""Remediation of the October 31, 2026 audit, package ``train-step`` (optimizer construction and the training step).

* train-step-1 (A099): the exact gradient-finiteness check reads every infinity norm in float64, so it equals the
  per-tensor verdict in every real dtype, and a finite float64 gradient above the float32 range takes its step.
* train-step-3 (A171): a binding clip of the token-gathered frame table scales only the step's input rows, bit for bit
  the dense clip; on the live and reference routes the frame gradient is zero outside those rows.
* baseline-generation-8 (A173): the conventional baseline's targets are validated on the host before the transfer,
  so its fused CE reads none back.
* gap-mixed-precision-routes-3 (A182): an fp16 step reads the loss scale from the host once.
* train-step-4 (A203): a silent accumulated step reads no microbatch count, the exclusion-count fallback builds no host
  tensor, and the fisher_group type counts call no ``bincount``.
* train-step-6 (A322): ``block_mlp_bias_weight_decay``.
* train-step-7 (A323): ``relational_content_weight_decay``; the content maps are flat along the per-head GL(d) orbit.
* train-step-8 (A324): ``adam_beta1``, ``adam_beta2`` and ``adam_eps`` reach every optimizer and the saved config.
* block-mlp-norm-9 (A332): the gaussian_template log variances and log weights take no decay.
* dead-code-2 (A335): ``PhiFisherStatisticsProvider`` takes no rotation weight.
* gap-mixed-precision-routes-8 (A338): an overflow found inside ``GradScaler.step`` is a non-finite skip.

Tiny CPU models only; every configuration comes from ``tests.profiles``.
"""
from __future__ import annotations

import inspect
import math
import warnings
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Tuple

import pytest
import torch

import vfe3.train as T
from tests.profiles import live_config, reference_k8_config, test_config
from vfe3.config import VFE3Config, config_from_serialized, serialize_config
from vfe3.model.model import VFEModel

CPU = torch.device("cpu")
NEW_FIELDS = ("block_mlp_bias_weight_decay", "relational_content_weight_decay", "adam_beta1", "adam_beta2",
              "adam_eps")


@pytest.fixture(autouse=True)
def _two_threads() -> None:
    torch.set_num_threads(2)


def _quiet(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(*args, **kwargs)


def _model(cfg: VFE3Config, seed: int = 0) -> VFEModel:
    torch.manual_seed(seed)
    return _quiet(VFEModel, cfg)


def _names(model: torch.nn.Module) -> Dict[int, str]:
    return {id(p): name for name, p in model.named_parameters()}


def _constant_schedule(optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LambdaLR:
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)


def _batch(cfg: VFE3Config, seed: int = 1, batch: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    shape = (batch, cfg.max_seq_len)
    return (torch.randint(0, cfg.vocab_size, shape, generator=g),
            torch.randint(0, cfg.vocab_size, shape, generator=g))


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    r"""The IEEE bit pattern of a float32 or float64 tensor, so -0.0 and +0.0 compare unequal."""
    return tensor.detach().contiguous().view({torch.float32: torch.int32, torch.float64: torch.int64}[tensor.dtype])


def _state_snapshot(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
    names = _names(model)
    params = {name: p.detach().clone() for name, p in model.named_parameters()}
    state = {names[id(p)]: {key: (value.detach().clone() if isinstance(value, torch.Tensor) else value)
                            for key, value in slot.items()}
             for p, slot in optimizer.state.items()}
    return {"params": params, "state": state}


def _assert_bitwise_equal(left: Dict[str, Any], right: Dict[str, Any]) -> None:
    assert left["params"].keys() == right["params"].keys()
    for name, value in left["params"].items():
        assert torch.equal(_bits(value), _bits(right["params"][name])), name
    assert left["state"].keys() == right["state"].keys()
    for name, slot in left["state"].items():
        assert slot.keys() == right["state"][name].keys(), name
        for key, value in slot.items():
            other = right["state"][name][key]
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                assert torch.equal(_bits(value), _bits(other)), (name, key)
            elif isinstance(value, torch.Tensor):
                assert torch.equal(value, other), (name, key)
            else:
                assert value == other, (name, key)


# ------------------------------------------------------------------------------------------------ train-step-1 (A099)

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_exact_finiteness_check_equals_the_per_tensor_verdict_in_every_real_dtype(dtype: torch.dtype) -> None:
    largest = torch.finfo(dtype).max
    for value in (0.0, 1.0, largest, -largest, math.inf, -math.inf, math.nan):
        gradient = torch.zeros(3, 2, dtype=dtype)
        gradient[1, 0] = value
        assert bool(T._gradients_finite([gradient])) == bool(T._gradient_isfinite(gradient)), (dtype, value)
        mixed = [gradient, torch.ones(4), torch.empty(0)]
        assert bool(T._gradients_finite(mixed)) == all(bool(T._gradient_isfinite(g)) for g in mixed), (dtype, value)


def test_a_float64_gradient_above_the_float32_range_is_finite() -> None:
    gradient = torch.full((5,), 1e39, dtype=torch.float64)
    assert bool(T._gradient_isfinite(gradient))
    assert bool(T._gradients_finite([gradient]))
    assert bool(T._gradients_finite([gradient, torch.randn(3)]))


def test_a_finite_float64_gradient_above_the_float32_range_takes_its_step() -> None:
    cfg = _quiet(test_config, decode_mode="linear")
    model = _model(cfg).double()
    optimizer = T.build_optimizer(model, cfg)
    scheduler = _constant_schedule(optimizer)
    weight = model.prior_bank.output_proj_weight
    handle = weight.register_hook(lambda grad: torch.full_like(grad, 1e39))
    before = weight.detach().clone()
    tokens, targets = _batch(cfg)
    status: Dict[str, Any] = {}
    try:
        _quiet(T.train_step, model, optimizer, scheduler, tokens, targets, grad_clip=1.0, status_out=status)
    finally:
        handle.remove()
    assert status["did_step"] is True and status["nonfinite_skip"] is False
    assert not torch.equal(weight.detach(), before) and bool(torch.isfinite(weight).all())


# ------------------------------------------------------------------------------------------------ train-step-3 (A171)

# B * N = 8 input positions with repeated types, fewer than the 13 frame rows, so the row form applies.
_ROW_TOKENS = torch.tensor([[1, 4, 4, 9], [1, 12, 0, 4]])
_ROW_TARGETS = torch.tensor([[4, 4, 9, 2], [12, 0, 4, 7]])


def _live_rows(**overrides: Any) -> VFE3Config:
    return _quiet(live_config, **{"vocab_size": 13, "gauge_transport": "on", **overrides})


def _reference_rows(**overrides: Any) -> VFE3Config:
    return _quiet(reference_k8_config, vocab_size=13, **overrides)


_ROW_PROFILES = {"live": _live_rows, "reference": _reference_rows}


@pytest.mark.parametrize("profile", sorted(_ROW_PROFILES))
def test_frame_gradient_is_zero_outside_the_input_rows(profile: str) -> None:
    cfg = _ROW_PROFILES[profile]()
    model = _model(cfg)
    torch.manual_seed(2)
    _, loss, _ = _quiet(model, _ROW_TOKENS, _ROW_TARGETS, return_decode_stats=True)
    loss.backward()
    gradient = model.prior_bank.phi_embed.grad
    outside = torch.ones(cfg.vocab_size, dtype=torch.bool)
    outside[_ROW_TOKENS.reshape(-1)] = False
    assert bool(outside.any()) and bool((~outside).any())
    # every bit, sign included: a nonzero row outside the batch would be left unclipped by the row form
    assert bool((_bits(gradient[outside]) == 0).all())
    assert bool((gradient[~outside] != 0).any())


@pytest.mark.parametrize("profile", sorted(_ROW_PROFILES))
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_row_clip_of_the_frame_table_is_the_dense_clip_bit_for_bit(profile: str, dtype: torch.dtype) -> None:
    cfg = _ROW_PROFILES[profile]()
    model = _model(cfg).to(dtype)
    torch.manual_seed(2)
    _, loss, _ = _quiet(model, _ROW_TOKENS, _ROW_TARGETS, return_decode_stats=True)
    loss.backward()
    table = model.prior_bank.phi_embed
    other = next(p for name, p in model.named_parameters() if "raw_score_q" in name)
    parameters = [table, other]
    original = {id(p): p.grad.detach().clone() for p in parameters}
    total = torch.nn.utils.get_total_norm([p.grad for p in parameters])
    max_norm = 1e-3 * float(total)
    torch.nn.utils.clip_grads_with_norm_(parameters, max_norm, total)
    dense = {id(p): p.grad.detach().clone() for p in parameters}
    for p in parameters:
        p.grad.copy_(original[id(p)])
    rows = T._token_frame_rows(model, _ROW_TOKENS)
    assert rows is not None and rows[0] is table
    T._clip_grads_with_norm_rows_(parameters, max_norm, total, *rows)
    for p in parameters:
        assert torch.equal(_bits(p.grad), _bits(dense[id(p)]))
    assert not torch.equal(dense[id(table)], original[id(table)])


@pytest.mark.parametrize("profile", sorted(_ROW_PROFILES))
def test_train_step_takes_the_row_clip_and_matches_the_dense_clip(profile: str,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    snapshots = []
    row_calls: List[int] = []
    for rows_enabled in (True, False):
        cfg = _ROW_PROFILES[profile]()
        model = _model(cfg)
        optimizer = T.build_optimizer(model, cfg)
        scheduler = _constant_schedule(optimizer)
        with monkeypatch.context() as patch:
            if rows_enabled:
                real = T._clip_grads_with_norm_rows_
                patch.setattr(T, "_clip_grads_with_norm_rows_",
                              lambda *args, **kwargs: (row_calls.append(1), real(*args, **kwargs))[1])
            else:
                patch.setattr(T, "_token_frame_rows", lambda *args, **kwargs: None)
            for step in range(2):
                torch.manual_seed(10 + step)
                status: Dict[str, Any] = {}
                _quiet(T.train_step, model, optimizer, scheduler, _ROW_TOKENS, _ROW_TARGETS, grad_clip=1e-3,
                       grad_clip_scope=cfg.grad_clip_scope, status_out=status)
                assert status["did_step"] is True
                assert status["clip_coefs"]["phi"] < 1.0                   # the frame clip binds
        snapshots.append(_state_snapshot(model, optimizer))
    assert len(row_calls) == 2                                              # one binding frame clip per step
    _assert_bitwise_equal(*snapshots)


def test_row_clip_applies_only_to_the_built_in_token_gathered_frame_table(monkeypatch: pytest.MonkeyPatch) -> None:
    import vfe3.model.prior_bank as prior_bank_module

    cfg = _live_rows()
    model = _model(cfg)
    rows = T._token_frame_rows(model, _ROW_TOKENS)
    assert rows is not None
    assert rows[0] is model.prior_bank.phi_embed and torch.equal(rows[1], _ROW_TOKENS.reshape(-1))
    # as many input positions as table rows: the dense multiply reads no more
    assert T._token_frame_rows(model, torch.zeros(4, 4, dtype=torch.long)) is None
    # an encoder outside the two built-in token gathers may read the table elsewhere
    registry = prior_bank_module.get_encode(model.prior_bank.encode_mode)
    monkeypatch.setattr(prior_bank_module, "get_encode", lambda mode: (lambda pb, ids: registry(pb, ids)))
    assert T._token_frame_rows(model, _ROW_TOKENS) is None
    monkeypatch.undo()
    # a frozen frame table has no gradient to clip
    frozen = _model(_live_rows(gauge_transport="frozen"))
    assert T._token_frame_rows(frozen, _ROW_TOKENS) is None
    # another model family
    assert T._token_frame_rows(torch.nn.Linear(2, 2), _ROW_TOKENS) is None


# ----------------------------------------------------------------------------------------- baseline-generation-8 (A173)

def _baseline() -> torch.nn.Module:
    from vfe3.baseline import BaselineConfig, BaselineTransformer

    base = _quiet(test_config, vocab_size=11, embed_dim=8, n_heads=2, max_seq_len=6, batch_size=2, n_layers=1,
                  decode_mode="linear", decode_unigram_prior=False, block_mlp_expansion=2)
    torch.manual_seed(0)
    return BaselineTransformer(BaselineConfig.matched_to(base))


def test_baseline_targets_are_validated_on_the_host_and_the_forward_reads_none(
        monkeypatch: pytest.MonkeyPatch) -> None:
    from vfe3.baseline import LinearUnigramDecoder

    model = _baseline()
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 0, 1]])
    targets = torch.tensor([[2, 3, -100, 5, 6, 7], [8, 9, 10, 0, 1, -100]])
    moved = T._transfer_targets(targets, model, CPU)
    assert moved is not targets and torch.equal(moved, targets)
    assert moved._vfe3_host_target_validation == (id(moved), moved._version, 11, -100, True)
    reads: List[int] = []
    fast: List[bool] = []
    original = LinearUnigramDecoder._validate_fused_ce_targets
    real_tolist = torch.Tensor.tolist

    def counted(self: Any, target_ids: torch.Tensor, **kwargs: Any) -> bool:
        fast.append("_vfe3_host_target_validation" in target_ids.__dict__)
        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "tolist", lambda tensor: (reads.append(1), real_tolist(tensor))[1])
            return original(self, target_ids, **kwargs)

    monkeypatch.setattr(LinearUnigramDecoder, "_validate_fused_ce_targets", counted)
    _, loss, stats = model(tokens, moved, return_decode_stats=True)
    assert fast == [True] and reads == []
    assert "_vfe3_host_target_validation" not in moved.__dict__
    assert int(stats.scored_tokens) == 10 and int(stats.excluded_tokens) == 2 and math.isfinite(float(loss.detach()))


def test_baseline_host_validation_refuses_an_out_of_range_target() -> None:
    model = _baseline()
    with pytest.raises(IndexError, match="Target 11"):
        T._transfer_targets(torch.tensor([[11, 1, 2, 3, 4, 5]]), model, CPU)


# ----------------------------------------------------------------------------------------- fp16 GradScaler (A182, A338)

def _fp16(**overrides: Any) -> Tuple[VFE3Config, VFEModel, torch.optim.Optimizer, Any, torch.amp.GradScaler]:
    values: Dict[str, Any] = dict(vocab_size=7, embed_dim=4, n_heads=2, max_seq_len=3, batch_size=2, n_layers=1,
                                  decode_mode="linear", decode_logits_amp=True, amp_dtype="fp16",
                                  grad_clip_scope="group", fp32_matmul_precision="highest", mu_init_std=0.3,
                                  phi_scale=0.05)
    values.update(overrides)
    cfg = _quiet(test_config, **values)
    model = _model(cfg)
    optimizer = T.build_optimizer(model, cfg)
    return cfg, model, optimizer, _constant_schedule(optimizer), torch.amp.GradScaler(device="cpu", enabled=True)


def test_an_fp16_step_reads_the_loss_scale_once(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg, model, optimizer, scheduler, scaler = _fp16()
    calls: List[int] = []
    real = torch.amp.GradScaler.get_scale
    monkeypatch.setattr(torch.amp.GradScaler, "get_scale", lambda self: (calls.append(1), real(self))[1])
    tokens, targets = _batch(cfg)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    _quiet(T.train_step, model, optimizer, scheduler, tokens, targets, grad_clip=1.0, grad_clip_scope="group",
           scaler=scaler)
    assert len(calls) == 1                                       # the post-update read only
    assert any(not torch.equal(before[name], p.detach()) for name, p in model.named_parameters())
    # an overflow still lowers the scale and reports the skipped update, with the same single read
    calls.clear()
    weight = model.prior_bank.output_proj_weight
    handle = weight.register_hook(lambda grad: torch.full_like(grad, math.inf))
    scale = scaler._scale.item()
    status: Dict[str, Any] = {}
    try:
        _quiet(T.train_step, model, optimizer, scheduler, tokens, targets, grad_clip=1.0, grad_clip_scope="group",
               scaler=scaler, status_out=status)
    finally:
        handle.remove()
    assert len(calls) == 1 and status["did_step"] is False and scaler._scale.item() < scale


def test_an_fp16_log_step_reports_the_pre_update_scale() -> None:
    cfg, model, optimizer, scheduler, scaler = _fp16()
    tokens, targets = _batch(cfg)
    scale = float(scaler.get_scale())
    metrics: Dict[str, Any] = {}
    _quiet(T.train_step, model, optimizer, scheduler, tokens, targets, grad_clip=1.0, grad_clip_scope="group",
           scaler=scaler, metrics_out=metrics)
    assert metrics["grad_scale"] == scale


@pytest.mark.parametrize("parameter", ["mu_embed", "output_proj_weight"])
def test_a_gradscaler_overflow_is_a_nonfinite_skip(parameter: str) -> None:
    # Under the table Fisher step mu_embed is never clipped, so its overflow is found inside GradScaler.step only.
    cfg, model, optimizer, scheduler, scaler = _fp16(m_table_update_mode="fisher")
    tokens, targets = _batch(cfg)
    status: Dict[str, Any] = {}
    _quiet(T.train_step, model, optimizer, scheduler, tokens, targets, grad_clip=1.0, grad_clip_scope="group",
           scaler=scaler, status_out=status)
    assert status["did_step"] is True and status["nonfinite_skip"] is False
    table = getattr(model.prior_bank, parameter)
    handle = table.register_hook(lambda grad: torch.full_like(grad, math.inf))
    scale = float(scaler.get_scale())
    before = _state_snapshot(model, optimizer)
    status = {}
    try:
        _quiet(T.train_step, model, optimizer, scheduler, tokens, targets, grad_clip=1.0, grad_clip_scope="group",
               scaler=scaler, status_out=status)
    finally:
        handle.remove()
    assert status["did_step"] is False and status["nonfinite_skip"] is True and status["loss_spike_skip"] is False
    assert float(scaler.get_scale()) < scale
    _assert_bitwise_equal(before, _state_snapshot(model, optimizer))


# ------------------------------------------------------------------------------------------------ train-step-4 (A203)

def test_a_silent_accumulated_step_reads_no_microbatch_count(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _quiet(test_config, vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1, m_phi_lr=0.0)
    model = _model(cfg)
    optimizer = T.build_optimizer(model, cfg)
    scheduler = _constant_schedule(optimizer)
    tokens, targets = _batch(cfg, seed=3, batch=4)
    targets = targets.clone()
    targets[0, :4] = -100                                       # microbatch 0 scores 12 targets, microbatch 1 scores 16
    inside: List[bool] = []
    reads: List[Tuple[str, Tuple[int, ...]]] = []

    def scoped(function: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            inside.append(True)
            try:
                return function(*args, **kwargs)
            finally:
                inside.pop()
        return wrapper

    monkeypatch.setattr(model, "forward", scoped(model.forward))
    monkeypatch.setattr(optimizer, "step", scoped(optimizer.step))
    for name in ("cpu", "tolist", "item"):
        real = getattr(torch.Tensor, name)

        def reader(self: torch.Tensor, *args: Any, _real: Callable[..., Any] = real, _name: str = name,
                   **kwargs: Any) -> Any:
            if not inside and self.dtype == torch.int64:
                reads.append((_name, tuple(self.shape)))
            return _real(self, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, name, reader)
    with pytest.warns(RuntimeWarning, match="non-CE regularizers"):
        T.train_step(model, optimizer, scheduler, tokens, targets, grad_clip=1.0, grad_accum_steps=2)
    monkeypatch.undo()
    assert reads == []
    # a log step still reports the spread of the scored-token counts
    metrics: Dict[str, Any] = {}
    _quiet(T.train_step, model, optimizer, scheduler, tokens, targets, grad_clip=1.0, grad_accum_steps=2,
           metrics_out=metrics)
    assert metrics["grad_accum_tok_spread"] == 4.0


def test_an_accumulated_fp16_step_reports_the_spread_and_the_pre_update_scale() -> None:
    cfg, model, optimizer, scheduler, _ = _fp16()
    scaler = torch.amp.GradScaler(device="cpu", init_scale=256.0, enabled=True)   # no fp16 overflow at this scale
    tokens, targets = _batch(cfg, seed=5)
    targets = targets.clone()
    targets[0, :2] = -100                                       # microbatch 0 scores 1 target, microbatch 1 scores 3
    scale = float(scaler.get_scale())
    metrics: Dict[str, Any] = {}
    status: Dict[str, Any] = {}
    with pytest.warns(RuntimeWarning, match="non-CE regularizers"):
        T.train_step(model, optimizer, scheduler, tokens, targets, grad_clip=1.0, grad_clip_scope="group",
                     grad_accum_steps=2, scaler=scaler, metrics_out=metrics, status_out=status)
    assert metrics["grad_accum_tok_spread"] == 2.0 and metrics["grad_scale"] == scale
    assert status["did_step"] is True and math.isfinite(status["ce"])


class _ScriptedCounts(torch.nn.Module):
    r"""A one-weight objective whose CE and scored count are each microbatch's first two token values."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.cfg = SimpleNamespace(m_phi_update_mode="adamw", phi_mstep_max_matrix_norm=None)

    def forward(self, tokens: torch.Tensor, targets: Any = None, *, estep_grad_out: Any = None,
                return_decode_stats: bool = False) -> Any:
        from vfe3.model.prior_bank import DecodeCEResult

        ce = self.weight * tokens[0, 0].to(self.weight.dtype)
        return None, ce, DecodeCEResult(ce=ce.detach(), scored_tokens=tokens[0, 1].to(dtype=torch.int64))


def test_the_fallback_transfer_carries_the_count_spread() -> None:
    # No optimizer parameter holds a gradient, so the loss resolves through the plain fallback transfer.
    model = _ScriptedCounts()
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.0)
    scheduler = _constant_schedule(optimizer)
    metrics: Dict[str, Any] = {}
    status: Dict[str, Any] = {}
    with pytest.warns(RuntimeWarning, match="non-CE regularizers"):
        loss = T.train_step(model, optimizer, scheduler, torch.tensor([[1, 1], [3, 3]]), torch.zeros(2, 2),
                            grad_clip=0.0, grad_accum_steps=2, metrics_out=metrics, status_out=status)
    assert loss == pytest.approx((1.0 * 1 + 3.0 * 3) / 4) and status["ce"] == pytest.approx(2.5)
    assert metrics["grad_accum_tok_spread"] == 2.0
    assert (metrics["expected_targets"], metrics["scored_targets"], metrics["excluded_targets"]) == (4, 4, 0)


def test_the_excluded_count_fallback_builds_no_host_tensor(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: Any, **kwargs: Any) -> torch.Tensor:
        raise AssertionError("new_tensor copies a host value to the device")

    monkeypatch.setattr(torch.Tensor, "new_tensor", refuse)
    excluded = T._excluded_token_count(
        SimpleNamespace(scored_tokens=torch.tensor(5, dtype=torch.int64), excluded_tokens=None), 12)
    assert excluded.dtype == torch.int64 and excluded.shape == () and int(excluded) == 7
    given = torch.tensor(3, dtype=torch.int64)
    assert T._excluded_token_count(SimpleNamespace(scored_tokens=torch.tensor(9), excluded_tokens=given), 12) is given


def _fisher_cfg(**overrides: Any) -> VFE3Config:
    values: Dict[str, Any] = dict(vocab_size=12, embed_dim=6, n_heads=2, max_seq_len=8, n_layers=1,
                                  gauge_group="block_glk", pos_phi="learned", m_phi_update_mode="fisher_group",
                                  transport_chart_max_norm=6.0, m_phi_lr=0.1)
    values.update(overrides)
    return _quiet(test_config, **values)


def test_fisher_group_type_counts_call_no_bincount(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _fisher_cfg()
    model = _model(cfg)
    provider = T.build_optimizer(model, cfg).phi_statistics_provider
    tokens = torch.tensor([[0, 1, 1, 5], [5, 5, 2, 1]])
    rows = torch.tensor([1, 5, 7, 0])
    expected = torch.bincount(tokens.reshape(-1), minlength=cfg.vocab_size)[rows].to(torch.float64)

    def refuse(*args: Any, **kwargs: Any) -> torch.Tensor:
        raise AssertionError("bincount sizes its output from the data, a host read on CUDA")

    monkeypatch.setattr(torch, "bincount", refuse)
    provider.set_batch(tokens, torch.tensor(7))
    statistics = provider(model.prior_bank.phi_embed, rows)
    assert statistics.count.dtype == torch.float64 and torch.equal(statistics.count, expected)


# ------------------------------------------------------------------------------------------------ dead-code-2 (A335)

def test_fisher_provider_takes_the_model_only_and_the_optimizer_holds_the_rotation_weight() -> None:
    assert list(inspect.signature(T.PhiFisherStatisticsProvider).parameters) == ["model"]
    cfg = _fisher_cfg(m_phi_fisher_rotation_weight=7.5)
    model = _model(cfg)
    optimizer = T.build_optimizer(model, cfg)
    assert isinstance(optimizer.phi_statistics_provider, T.PhiFisherStatisticsProvider)
    assert not hasattr(optimizer.phi_statistics_provider, "_rotation_weight")
    assert optimizer._phi_fisher_rotation_weight == 7.5


# ------------------------------------------------------------------------------------------------ train-step-6 (A322)

def _block_mlp_groups(model: VFEModel, optimizer: torch.optim.Optimizer) -> Dict[str, Dict[str, Any]]:
    names = _names(model)
    return {names[id(p)]: group for group in optimizer.param_groups for p in group["params"]
            if names[id(p)].startswith("block_mlps.")}


def test_block_mlp_bias_decay_unset_keeps_the_one_block_mlp_group() -> None:
    cfg = _quiet(test_config)
    assert cfg.block_mlp_bias_weight_decay is None
    model = _model(cfg)
    optimizer = T.build_optimizer(model, cfg)
    groups = _block_mlp_groups(model, optimizer)
    assert {name.rsplit(".", 2)[-2] + "." + name.rsplit(".", 1)[-1] for name in groups} == {
        "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"}
    assert len({id(group) for group in groups.values()}) == 1
    group = next(iter(groups.values()))
    assert group["lr_aux_role"] == "block_mlp" and group["weight_decay"] == cfg.weight_decay


@pytest.mark.parametrize("tie", [False, True])
def test_block_mlp_bias_decay_moves_the_linear_biases_to_their_own_group(tie: bool) -> None:
    cfg = _quiet(test_config, n_layers=2, block_mlp_tie_layers=tie, block_mlp_bias_weight_decay=0.0)
    model = _model(cfg)
    optimizer = T.build_optimizer(model, cfg)
    groups = _block_mlp_groups(model, optimizer)
    biases = {name: group for name, group in groups.items() if name.endswith(".bias")}
    weights = {name: group for name, group in groups.items() if name.endswith(".weight")}
    assert len(biases) == len(weights) == (2 if tie else 4)
    (bias_group,) = {id(group): group for group in biases.values()}.values()
    (weight_group,) = {id(group): group for group in weights.values()}.values()
    assert bias_group is not weight_group
    assert bias_group["weight_decay"] == 0.0 and "lr_aux_role" not in bias_group and bias_group["role"] == "mu"
    assert weight_group["weight_decay"] == cfg.weight_decay and weight_group["lr_aux_role"] == "block_mlp"
    assert bias_group["lr"] == weight_group["lr"] == cfg.resolved_hyperparameter("m_block_mlp_lr")
    assert len(bias_group["params"]) == (2 if tie else 4)
    # one step with zero gradients: decoupled decay alone scales the weights, the biases do not move
    named = dict(model.named_parameters())
    before = {name: named[name].detach().clone() for name in groups}
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    optimizer.step()
    for name in weights:
        factor = 1.0 - weight_group["lr"] * cfg.weight_decay
        torch.testing.assert_close(named[name].detach(), before[name] * factor, rtol=1e-6, atol=0.0)
    for name in biases:
        assert torch.equal(named[name].detach(), before[name])


@pytest.mark.parametrize("value", [-0.1, math.nan, math.inf, True])
def test_block_mlp_bias_weight_decay_is_validated(value: Any) -> None:
    with pytest.raises(ValueError, match="block_mlp_bias_weight_decay"):
        _quiet(test_config, block_mlp_bias_weight_decay=value)


def test_block_mlp_bias_weight_decay_is_inert_without_linear_biases() -> None:
    assert "block_mlp_bias_weight_decay" not in _quiet(test_config).inactive_fields()
    assert "block_mlp_bias_weight_decay" in _quiet(test_config, use_block_mlp=False).inactive_fields()
    assert "block_mlp_bias_weight_decay" in _quiet(test_config, block_mlp_mode="gaussian_template").inactive_fields()
    unset = _quiet(test_config, weight_decay=0.03)
    assert unset.resolved_hyperparameter("block_mlp_bias_weight_decay") == 0.03
    assert unset.inherits_hyperparameter("block_mlp_bias_weight_decay", "weight_decay")
    exempt = _quiet(test_config, weight_decay=0.03, block_mlp_bias_weight_decay=0.0)
    assert exempt.resolved_hyperparameter("block_mlp_bias_weight_decay") == 0.0


# ---------------------------------------------------------------------------------------------- block-mlp-norm-9 (A332)

def test_gaussian_template_log_variances_and_log_weights_take_no_decay() -> None:
    cfg = _quiet(test_config, block_mlp_mode="gaussian_template", n_layers=2)
    model = _model(cfg)
    optimizer = T.build_optimizer(model, cfg)
    groups = _block_mlp_groups(model, optimizer)
    leaves = {name.rsplit(".", 1)[-1] for name in groups}
    assert leaves == {"nu", "log_lambda", "log_pi", "values"}
    for name, group in groups.items():
        expected = 0.0 if name.rsplit(".", 1)[-1] in ("log_lambda", "log_pi") else cfg.weight_decay
        assert group["weight_decay"] == expected, name


# ------------------------------------------------------------------------------------------------ train-step-7 (A323)

def test_content_maps_are_flat_along_the_per_head_gl_orbit() -> None:
    r"""The kernel readout's loss is invariant under :math:`(T_h, O_h) \to (D_h T_h, O_h D_h^{-1})` (exp chart)."""
    cfg = _quiet(reference_k8_config)
    assert cfg.content_link_chart == "exponential" and cfg.predictive_readout == "kernel"
    model = _model(cfg).double()
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for ops in model.relational_operators_by_layer:
            ops.T.add_(0.2 * torch.randn(ops.T.shape, generator=g, dtype=torch.float64))
            ops.O.add_(0.2 * torch.randn(ops.O.shape, generator=g, dtype=torch.float64))
    tokens, targets = _batch(cfg, seed=4)

    def loss() -> torch.Tensor:
        return _quiet(model, tokens, targets, return_decode_stats=True)[1]

    base = loss()
    base.backward()
    ops = model.relational_operators
    H, d, _ = ops.T.shape
    for _ in range(3):
        A = torch.randn(H, d, d, generator=g, dtype=torch.float64)            # (H, d, d) orbit generator
        dT, dO = A @ ops.T.detach(), -ops.O.detach() @ A                      # (H, d, K), (H, K, d)
        directional = float((ops.T.grad * dT).sum() + (ops.O.grad * dO).sum())
        scale = float(ops.T.grad.norm() * dT.norm() + ops.O.grad.norm() * dO.norm())
        assert scale > 1e-3 and abs(directional) <= 1e-12 * scale
    with torch.no_grad():
        D = 4.0 * torch.eye(d, dtype=torch.float64) + 0.3 * torch.randn(H, d, d, generator=g, dtype=torch.float64)
        ops.T.copy_(D @ ops.T)
        ops.O.copy_(ops.O @ torch.linalg.inv(D))
        assert abs(float(loss()) - float(base)) <= 1e-12 * abs(float(base))


def test_content_weight_decay_off_keeps_the_single_undecayed_content_group() -> None:
    cfg = _quiet(test_config)
    assert cfg.relational_content_weight_decay == 0.0
    optimizer = T.build_optimizer(_model(cfg), cfg)
    groups = [group for group in optimizer.param_groups if group.get("lr_aux_role") == "relational_content"]
    assert len(groups) == 1 and groups[0]["weight_decay"] == 0.0


def test_content_weight_decay_reaches_the_content_maps() -> None:
    cfg = _quiet(test_config, n_layers=2, relational_content_weight_decay=0.5)
    model = _model(cfg)
    optimizer = T.build_optimizer(model, cfg)
    names = _names(model)
    decayed = [group for group in optimizer.param_groups if group["weight_decay"] == 0.5]
    assert len(decayed) == 1 and decayed[0]["lr_aux_role"] == "relational_content"
    assert sorted(names[id(p)].rsplit(".", 1)[-1] for p in decayed[0]["params"]) == ["O", "O", "T", "T"]
    named = dict(model.named_parameters())
    before = {names[id(p)]: p.detach().clone() for p in decayed[0]["params"]}
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    optimizer.step()
    factor = 1.0 - decayed[0]["lr"] * 0.5
    for name, value in before.items():
        torch.testing.assert_close(named[name].detach(), value * factor, rtol=1e-6, atol=0.0)


def test_content_weight_decay_is_validated() -> None:
    with pytest.raises(ValueError, match="relational_content_weight_decay"):
        _quiet(test_config, relational_content_weight_decay=-0.1)


# ------------------------------------------------------------------------------------------------ train-step-8 (A324)

_OPTIMIZER_ROUTES: Dict[str, Dict[str, Any]] = {
    "adamw": {},
    "table_fisher": dict(m_table_update_mode="fisher", decode_mode="linear"),
    "row_lazy": dict(m_phi_sparse_rows=True),
    "row_lazy_table_fisher": dict(m_phi_sparse_rows=True, m_table_update_mode="fisher", decode_mode="linear"),
    "manifold": dict(m_phi_update_mode="pullback_group", pos_phi="none"),
    "manifold_table_fisher": dict(m_phi_update_mode="pullback_group", m_table_update_mode="fisher",
                                  decode_mode="linear", pos_phi="none"),
}


def test_default_adam_constants_are_the_torch_defaults() -> None:
    reference = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))]).defaults
    cfg = _quiet(test_config)
    assert (cfg.adam_beta1, cfg.adam_beta2) == reference["betas"] and cfg.adam_eps == reference["eps"]
    optimizer = T.build_optimizer(_model(cfg), cfg)
    assert optimizer.defaults["betas"] == reference["betas"] and optimizer.defaults["eps"] == reference["eps"]
    assert all(group["betas"] == reference["betas"] and group["eps"] == reference["eps"]
               for group in optimizer.param_groups)


@pytest.mark.parametrize("route", sorted(_OPTIMIZER_ROUTES))
def test_adam_constants_reach_every_optimizer(route: str) -> None:
    cfg = _quiet(test_config, adam_beta1=0.8, adam_beta2=0.95, adam_eps=1e-6, **_OPTIMIZER_ROUTES[route])
    optimizer = T.build_optimizer(_model(cfg), cfg)
    assert optimizer.defaults["betas"] == (0.8, 0.95) and optimizer.defaults["eps"] == 1e-6
    assert all(group["betas"] == (0.8, 0.95) and group["eps"] == 1e-6 for group in optimizer.param_groups)


def test_adam_constants_change_the_adamw_step() -> None:
    steps = []
    for beta2 in (0.999, 0.5):
        cfg = _quiet(test_config, adam_beta2=beta2)
        model = _model(cfg)
        optimizer = T.build_optimizer(model, cfg)
        g = torch.Generator().manual_seed(7)
        for _ in range(2):
            for p in model.parameters():
                p.grad = torch.randn(p.shape, generator=g)
            optimizer.step()
        steps.append(model.prior_bank.mu_embed.detach().clone())
    assert not torch.equal(*steps)


def test_new_fields_are_recorded_and_older_records_reload_the_introduction_values() -> None:
    cfg = _quiet(test_config, adam_beta1=0.8, adam_beta2=0.95, adam_eps=1e-6, block_mlp_bias_weight_decay=0.0,
                 relational_content_weight_decay=0.25)
    payload = serialize_config(cfg)
    assert [payload[name] for name in NEW_FIELDS] == [0.0, 0.25, 0.8, 0.95, 1e-6]
    reloaded = _quiet(config_from_serialized, payload, source="new")
    assert [getattr(reloaded, name) for name in NEW_FIELDS] == [0.0, 0.25, 0.8, 0.95, 1e-6]
    older = {name: value for name, value in serialize_config(_quiet(test_config)).items() if name not in NEW_FIELDS}
    reloaded = _quiet(config_from_serialized, older, source="old")
    assert [getattr(reloaded, name) for name in NEW_FIELDS] == [None, 0.0, 0.9, 0.999, 1e-8]


@pytest.mark.parametrize("overrides", [dict(adam_beta1=1.0), dict(adam_beta1=-0.1), dict(adam_beta2=1.0),
                                       dict(adam_beta2=math.nan), dict(adam_eps=0.0), dict(adam_eps=math.inf),
                                       dict(adam_eps=-1e-8), dict(adam_beta1=True)])
def test_adam_constants_are_validated(overrides: Dict[str, Any]) -> None:
    (name,) = overrides
    with pytest.raises(ValueError, match=name):
        _quiet(test_config, **overrides)
