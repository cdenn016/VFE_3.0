"""Outer-objective covector and training-boundary proofs for phi group descent."""

from tests.profiles import test_config

import copy
import types

import pytest
import torch
import torch.nn.functional as F

from vfe3.config import VFE3Config
import vfe3.gauge_optim as gauge_optim
from vfe3.model.model import VFEModel
from vfe3.train import build_optimizer, train_step


def _outer_cfg(**overrides: object) -> VFE3Config:
    values: dict[str, object] = {
        "vocab_size": 9,
        "embed_dim": 4,
        "n_heads": 2,
        "max_seq_len": 4,
        "n_layers": 1,
        "n_e_steps": 1,
        "gauge_group": "block_glk",
        "pos_phi": "none",
        "mass_phi": 0.0,
        "mstep_self_coupling_weight": 0.0,
        "seed": 19,
        "mu_init_std": 0.02,
        "sigma_init": 1.0,
        "phi_scale": 0.01,
        "lambda_alpha_mode": "state_dependent",
        "norm_type_block": "layernorm",
        "layernorm_affine": True,
        "skip_belief_sigma_update": True,
        "beta_attention_prior": "causal_alibi",
    }
    values.update(overrides)
    return test_config(**values)


def _fixed_batch() -> tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.long)
    targets = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long)
    return tokens, targets


def _scheduler(optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LambdaLR:
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)


def _seeded_active_direction(
    parameter: torch.Tensor,
    tokens:    torch.Tensor,
) -> torch.Tensor:
    direction = torch.zeros_like(parameter)
    active_rows = torch.unique(tokens)
    direction[active_rows] = torch.randn(
        (active_rows.numel(), parameter.shape[-1]),
        generator=torch.Generator().manual_seed(406),
        dtype=parameter.dtype,
    )
    return direction / torch.linalg.vector_norm(direction)


def _central_differences(
    model:     VFEModel,
    parameter: torch.nn.Parameter,
    direction: torch.Tensor,
    tokens:    torch.Tensor,
    targets:   torch.Tensor,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    original = parameter.detach().clone()
    loss_estimates: list[torch.Tensor] = []
    ce_estimates: list[torch.Tensor] = []
    try:
        for step in (1.0e-2, 3.0e-3, 1.0e-3):
            with torch.no_grad():
                parameter.copy_(original + step * direction)
            _, plus_loss, plus_ce = model(tokens, targets)
            with torch.no_grad():
                parameter.copy_(original - step * direction)
            _, minus_loss, minus_ce = model(tokens, targets)
            loss_estimates.append((plus_loss.detach() - minus_loss.detach()) / (2.0 * step))
            ce_estimates.append((plus_ce - minus_ce) / (2.0 * step))
    finally:
        with torch.no_grad():
            parameter.copy_(original)
    return loss_estimates, ce_estimates


def test_returned_outer_ce_phi_covector_matches_central_differences(record_property) -> None:
    torch.manual_seed(7)
    model = VFEModel(_outer_cfg(n_e_steps=2))
    tokens, targets = _fixed_batch()
    parameter = model.prior_bank.phi_embed
    direction = _seeded_active_direction(parameter, tokens)

    torch.manual_seed(0)
    _, loss, ce = model(tokens, targets)
    assert torch.equal(loss.detach(), ce)
    (gradient,) = torch.autograd.grad(loss, parameter)
    autograd_directional = torch.sum(gradient * direction).detach()
    finite_differences, _ = _central_differences(
        model, parameter, direction, tokens, targets
    )

    scale = autograd_directional.abs().clamp_min(1.0e-8)
    relative_errors = torch.stack(
        [(estimate - autograd_directional).abs() / scale for estimate in finite_differences]
    )
    record_property("autograd_directional", float(autograd_directional))
    for step, estimate, error in zip(
        (1.0e-2, 3.0e-3, 1.0e-3), finite_differences, relative_errors
    ):
        record_property(f"fd_h_{step:g}", float(estimate))
        record_property(f"relative_error_h_{step:g}", float(error))
    for step, estimate in zip((1e-2, 3e-3, 1e-3), finite_differences):
        rounding = 2 * torch.finfo(ce.dtype).eps * ce.abs() / step
        assert (estimate - autograd_directional).abs() <= 5e-3 * scale + rounding
    # Float32 loss subtraction has O(eps * loss / h) rounding error.
    rounding = 2 * torch.finfo(ce.dtype).eps * ce.abs() / 1e-3
    assert float(torch.stack(finite_differences).sub(finite_differences[-1]).abs().max()) <= float(rounding)


def test_completed_outer_loss_not_reconstructed_ce_supplies_phi_covector(record_property) -> None:
    torch.manual_seed(7)
    model = VFEModel(_outer_cfg(mass_phi=3.0, n_e_steps=2))
    tokens, targets = _fixed_batch()
    parameter = model.prior_bank.phi_embed
    direction = _seeded_active_direction(parameter, tokens)
    with torch.no_grad():
        parameter.add_(0.08 * direction)

    torch.manual_seed(0)
    _, loss, ce = model(tokens, targets)
    assert float(loss.detach()) > float(ce)
    (gradient,) = torch.autograd.grad(loss, parameter)
    autograd_directional = torch.sum(gradient * direction).detach()
    loss_estimates, ce_estimates = _central_differences(
        model, parameter, direction, tokens, targets
    )
    loss_fd = loss_estimates[1]
    ce_fd = ce_estimates[1]
    relative_error = (
        (loss_fd - autograd_directional).abs()
        / autograd_directional.abs().clamp_min(1.0e-8)
    )
    record_property("completed_loss_autograd_directional", float(autograd_directional))
    record_property("completed_loss_fd_h_0.003", float(loss_fd))
    record_property("reconstructed_ce_fd_h_0.003", float(ce_fd))
    record_property("completed_loss_relative_error", float(relative_error))
    assert float(relative_error) <= 5.0e-3
    assert float((loss_fd - ce_fd).abs()) >= 0.5 * float(autograd_directional.abs())


def _clip_reference_roles(optimizer: torch.optim.Optimizer, clip: float) -> float:
    """Clip each reference role independently and return the pre-clip phi norm."""
    role_parameters: dict[str, list[torch.Tensor]] = {}
    for group in optimizer.param_groups:
        role_parameters.setdefault(group.get("role", "other"), []).extend(group["params"])
    norms = {role: float(torch.nn.utils.clip_grad_norm_(parameters, clip))
             for role, parameters in role_parameters.items()}
    return norms["phi"]


def _processed_regularized_phi_covector(
    model:    VFEModel,
    tokens:   torch.Tensor,
    targets:  torch.Tensor,
    objective: str,
    clip:      float,
) -> tuple[torch.Tensor, float]:
    optimizer = build_optimizer(model, model.cfg)
    scaler = torch.amp.GradScaler(device="cpu", enabled=True, init_scale=128.0)
    optimizer.zero_grad(set_to_none=True)
    torch.manual_seed(0)
    logits, completed_loss, returned_ce = model(tokens, targets)
    if objective == "completed_loss":
        selected = completed_loss
    else:
        assert objective == "reconstructed_ce"
        if logits is None:
            torch.manual_seed(0)
            logits = model(tokens)
        flat_logits = logits.reshape(-1, model.cfg.vocab_size)
        flat_targets = targets.reshape(-1)
        n_valid = (flat_targets != -100).sum().clamp_min(1)
        selected = F.cross_entropy(
            flat_logits,
            flat_targets,
            ignore_index=-100,
            reduction="sum",
        ) / n_valid
        torch.testing.assert_close(selected.detach(), returned_ce, rtol=1e-6, atol=1e-7)
    scaler.scale(selected).backward()
    scaler.unscale_(optimizer)
    pre_clip_norm = _clip_reference_roles(optimizer, clip)
    gradient = model.prior_bank.phi_embed.grad.detach().clone()
    active = gradient.abs().sum(dim=-1) > 0
    return gradient[active].double(), pre_clip_norm


def _production_regularized_covectors(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    cfg = _pullback_cfg(
        mass_phi=3.0,
        decode_mode="diagonal_chunked",
        n_e_steps=2,
        amp_dtype="fp16",
    )
    torch.manual_seed(53)
    template = VFEModel(cfg)
    tokens, targets = _fixed_batch()
    direction = _seeded_active_direction(template.prior_bank.phi_embed, tokens)
    with torch.no_grad():
        template.prior_bank.phi_embed.add_(0.08 * direction)
    completed_model = copy.deepcopy(template)
    ce_model = copy.deepcopy(template)
    actual_model = copy.deepcopy(template)
    clip = 0.01
    completed, completed_pre_clip = _processed_regularized_phi_covector(
        completed_model,
        tokens,
        targets,
        "completed_loss",
        clip,
    )
    reconstructed_ce, ce_pre_clip = _processed_regularized_phi_covector(
        ce_model,
        tokens,
        targets,
        "reconstructed_ce",
        clip,
    )

    received: list[torch.Tensor] = []
    original_direction = gauge_optim.pullback_group_direction

    def _direction_spy(grad_phi, *args, **kwargs):
        received.append(grad_phi.detach().clone())
        return original_direction(grad_phi, *args, **kwargs)

    monkeypatch.setattr(gauge_optim, "pullback_group_direction", _direction_spy)
    actual_optimizer = build_optimizer(actual_model, actual_model.cfg)
    actual_scaler = torch.amp.GradScaler(device="cpu", enabled=True, init_scale=128.0)
    torch.manual_seed(0)
    train_step(
        actual_model,
        actual_optimizer,
        _scheduler(actual_optimizer),
        tokens,
        targets,
        grad_clip=clip,
        scaler=actual_scaler,
    )
    assert len(received) == 1
    return received[0], completed, reconstructed_ce, completed_pre_clip, ce_pre_clip


def test_train_step_consumes_completed_regularized_loss_covector_at_geometry_seam(
    monkeypatch: pytest.MonkeyPatch,
    record_property,
) -> None:
    received, completed, reconstructed_ce, completed_pre_clip, ce_pre_clip = (
        _production_regularized_covectors(monkeypatch)
    )
    torch.testing.assert_close(received, completed, rtol=0.0, atol=0.0)
    separation = torch.linalg.vector_norm(received - reconstructed_ce)
    record_property("regularized_received_covector_norm", float(torch.linalg.vector_norm(received)))
    record_property("regularized_ce_covector_norm", float(torch.linalg.vector_norm(reconstructed_ce)))
    record_property("regularized_covector_separation", float(separation))
    record_property("regularized_completed_pre_clip_norm", completed_pre_clip)
    record_property("regularized_ce_pre_clip_norm", ce_pre_clip)
    assert separation >= (
        0.1 * torch.linalg.vector_norm(received)
    )


def _pullback_cfg(**overrides: object) -> VFE3Config:
    values: dict[str, object] = {
        "m_phi_update_mode": "pullback_group",
        "m_phi_group_trust_radius": 0.1,
        "transport_chart_max_norm": 6.0,
        "m_phi_lr": 0.02,
    }
    values.update(overrides)
    return _outer_cfg(**values)


def test_geometry_receives_exact_post_unscale_post_clip_covector(
    monkeypatch: pytest.MonkeyPatch,
    record_property,
) -> None:
    cfg = _pullback_cfg(amp_dtype="fp16")
    torch.manual_seed(31)
    reference = VFEModel(cfg)
    actual = copy.deepcopy(reference)
    tokens, targets = _fixed_batch()
    clip = 0.001

    reference_optimizer = build_optimizer(reference, cfg)
    reference_scaler = torch.amp.GradScaler(device="cpu", enabled=True, init_scale=128.0)
    reference_optimizer.zero_grad(set_to_none=True)
    reference_loss = reference(tokens, targets)[1]
    reference_scaler.scale(reference_loss).backward()
    reference_scaler.unscale_(reference_optimizer)
    pre_clip_norm = _clip_reference_roles(reference_optimizer, clip)
    assert pre_clip_norm > clip
    expected_full = reference.prior_bank.phi_embed.grad.detach().clone()
    active = expected_full.abs().sum(dim=-1) > 0
    expected = expected_full[active].double()

    received: list[torch.Tensor] = []
    original_direction = gauge_optim.pullback_group_direction

    def _direction_spy(grad_phi, *args, **kwargs):
        received.append(grad_phi.detach().clone())
        return original_direction(grad_phi, *args, **kwargs)

    monkeypatch.setattr(gauge_optim, "pullback_group_direction", _direction_spy)
    actual_optimizer = build_optimizer(actual, cfg)
    actual_scaler = torch.amp.GradScaler(device="cpu", enabled=True, init_scale=128.0)
    train_step(
        actual,
        actual_optimizer,
        _scheduler(actual_optimizer),
        tokens,
        targets,
        grad_clip=clip,
        scaler=actual_scaler,
    )

    assert len(received) == 1
    torch.testing.assert_close(received[0], expected, rtol=0.0, atol=0.0)
    record_property("pre_clip_phi_norm", pre_clip_norm)
    record_property("processed_phi_covector_norm", float(torch.linalg.vector_norm(expected)))
    assert not torch.isclose(torch.linalg.vector_norm(received[0]), torch.tensor(1.0, dtype=torch.float64))


def test_pullback_group_gradient_accumulation_uses_accumulated_covector(
    monkeypatch: pytest.MonkeyPatch,
    record_property,
) -> None:
    cfg = _pullback_cfg()
    torch.manual_seed(37)
    full_model = VFEModel(cfg)
    accumulated_model = copy.deepcopy(full_model)
    tokens = torch.tensor(
        [[0, 1, 2, 3], [4, 5, 6, 7], [0, 2, 4, 6], [1, 3, 5, 7]],
        dtype=torch.long,
    )
    targets = torch.tensor(
        [[1, 2, 3, 4], [5, 6, 7, 8], [2, 4, 6, 8], [3, 5, 7, 0]],
        dtype=torch.long,
    )
    captures: list[tuple[torch.Tensor, torch.Tensor]] = []
    original_stage = gauge_optim.stage_pullback_group_candidate

    def _stage_spy(grad_phi, *args, **kwargs):
        candidate = original_stage(grad_phi, *args, **kwargs)
        captures.append((
            grad_phi.detach().clone(),
            candidate.candidate_phi.detach().clone(),
        ))
        return candidate

    monkeypatch.setattr(gauge_optim, "stage_pullback_group_candidate", _stage_spy)
    full_optimizer = build_optimizer(full_model, cfg)
    train_step(
        full_model,
        full_optimizer,
        _scheduler(full_optimizer),
        tokens,
        targets,
        grad_clip=None,
    )
    accumulated_optimizer = build_optimizer(accumulated_model, cfg)
    train_step(
        accumulated_model,
        accumulated_optimizer,
        _scheduler(accumulated_optimizer),
        tokens,
        targets,
        grad_clip=None,
        grad_accum_steps=2,
    )

    assert len(captures) == 2
    full_grad, full_candidate = captures[0]
    accumulated_grad, accumulated_candidate = captures[1]
    grad_error = torch.max((full_grad - accumulated_grad).abs())
    candidate_error = torch.max((full_candidate - accumulated_candidate).abs())
    record_property("accumulated_covector_max_abs_error", float(grad_error))
    record_property("accumulated_candidate_max_abs_error", float(candidate_error))
    torch.testing.assert_close(accumulated_grad, full_grad, rtol=2.0e-5, atol=2.0e-7)
    torch.testing.assert_close(
        accumulated_candidate,
        full_candidate,
        rtol=2.0e-5,
        atol=2.0e-7,
    )


@pytest.mark.parametrize("rejection", ["nonfinite_gradient", "nonfinite_loss"])
def test_rejected_training_attempt_preserves_every_phi_factor_and_clock(
    monkeypatch: pytest.MonkeyPatch,
    rejection: str,
) -> None:
    cfg = _pullback_cfg(pos_phi="none")
    torch.manual_seed(41)
    model = VFEModel(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = _scheduler(optimizer)
    tokens, targets = _fixed_batch()
    factors = (
        model.prior_bank.phi_embed,
    )
    before = [parameter.detach().clone() for parameter in factors]
    scheduler_epoch = scheduler.last_epoch
    staging_calls: list[int] = []
    original_stage = gauge_optim.stage_pullback_group_candidate

    def _stage_spy(*args, **kwargs):
        staging_calls.append(1)
        return original_stage(*args, **kwargs)

    monkeypatch.setattr(gauge_optim, "stage_pullback_group_candidate", _stage_spy)
    if rejection == "nonfinite_gradient":
        model.prior_bank.output_proj_weight.register_hook(
            lambda gradient: torch.full_like(gradient, float("inf"))
        )
    else:
        original_forward = model.forward

        def _nonfinite_forward(
            self,
            token_ids,
            target_ids=None,

            *,
            estep_grad_out=None,
            return_decode_stats=False,
        ):
            logits, loss, decode_result = original_forward(
                token_ids,
                target_ids,
                estep_grad_out=estep_grad_out,
                return_decode_stats=return_decode_stats,
            )
            return logits, loss * 0.0 + loss.new_tensor(float("inf")), decode_result

        model.forward = types.MethodType(_nonfinite_forward, model)

    status: dict[str, bool] = {}
    train_step(
        model,
        optimizer,
        scheduler,
        tokens,
        targets,
        grad_clip=1.0,
        status_out=status,
    )

    assert status["did_step"] is False and status["nonfinite_skip"] is True
    assert staging_calls == []
    assert scheduler.last_epoch == scheduler_epoch
    for parameter, expected in zip(factors, before):
        assert torch.equal(parameter, expected)


def test_pullback_group_train_step_bypasses_generic_phi_projector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vfe3.train as train_module

    cfg = _pullback_cfg(
        phi_mstep_max_matrix_norm=1.0,
        transport_chart_max_norm=2.0,
    )
    torch.manual_seed(43)
    model = VFEModel(cfg)
    optimizer = build_optimizer(model, cfg)
    tokens, targets = _fixed_batch()

    def _unexpected_projector(*args, **kwargs):
        raise AssertionError("pullback-group steps must not run the generic phi projector")

    monkeypatch.setattr(train_module, "project_phi_parameter_rows_", _unexpected_projector)
    status: dict[str, bool] = {}
    train_step(
        model,
        optimizer,
        _scheduler(optimizer),
        tokens,
        targets,
        status_out=status,
    )
    assert status["did_step"] is True and status["nonfinite_skip"] is False
