r"""fp16 training GradScaler (close the silent-mistrain footgun). Spec:
docs/superpowers/specs/2026-06-08-fp16-gradscaler-design.md.
"""

from tests.profiles import test_config
import pytest
import torch

from vfe3.config import VFE3Config
import vfe3.gauge_optim as gauge_optim
from vfe3.model.model import VFEModel
from vfe3.train import build_optimizer, train, train_step


def _tiny_cfg(**overrides) -> VFE3Config:
    base = dict(vocab_size=8, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=1,
                mu_init_std=0.02, sigma_init=1.0, phi_scale=0.01, lambda_alpha_mode="state_dependent",
                e_q_mu_lr=0.05, e_q_sigma_lr=0.02)
    base.update(overrides)
    return test_config(**base)


def _sched(opt):
    return torch.optim.lr_scheduler.LambdaLR(opt, lambda _step: 1.0)


def _batch(cfg, b=2, n=4):
    tok = torch.randint(0, cfg.vocab_size, (b, n))
    tgt = torch.randint(0, cfg.vocab_size, (b, n))
    return tok, tgt


def test_gradscaler_disabled_is_byte_identical_to_unscaled_step():
    # amp_dtype=None -> scaler enabled=False -> the wired train_step must produce the SAME
    # parameter update as the original unscaled loss.backward()/optimizer.step() path.
    cfg = _tiny_cfg()  # amp_dtype defaults to None
    torch.manual_seed(0); mA = VFEModel(cfg)
    torch.manual_seed(0); mB = VFEModel(cfg)
    tok, tgt = _batch(cfg)

    optA = build_optimizer(mA, cfg); schA = _sched(optA)
    train_step(mA, optA, schA, tok, tgt, grad_clip=1.0)

    optB = build_optimizer(mB, cfg)
    optB.zero_grad(set_to_none=True)
    _, lossB, _ = mB(tok, tgt)
    lossB.backward()
    role_parameters = {}
    for group in optB.param_groups:
        role_parameters.setdefault(group.get("role", "other"), []).extend(group["params"])
    for parameters in role_parameters.values():
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    optB.step()

    for (na, pa), (nb, pb) in zip(mA.named_parameters(), mB.named_parameters()):
        assert torch.equal(pa, pb), f"param {na} diverged from the unscaled reference"


def test_fp16_forward_keeps_logits_and_sigma_finite():
    # The SPD/sigma islands must stay fp32 under amp_dtype='fp16' (no Cholesky/eigh NaN).
    torch.manual_seed(0)
    m = VFEModel(_tiny_cfg(amp_dtype="fp16"))
    tok = torch.randint(0, m.cfg.vocab_size, (2, 4))
    lg = m(tok)
    assert torch.isfinite(lg).all()


def _run_one_fp16_step(cfg):
    torch.manual_seed(0)
    m = VFEModel(cfg)
    opt = build_optimizer(m, cfg); sch = _sched(opt)
    scaler = torch.amp.GradScaler(device="cpu", enabled=True)
    before = {n: p.detach().clone() for n, p in m.named_parameters() if p.requires_grad}
    tok, tgt = _batch(cfg)
    train_step(m, opt, sch, tok, tgt, grad_clip=1.0, scaler=scaler)
    moved = any(not torch.equal(before[n], p) for n, p in m.named_parameters()
                if p.requires_grad and n in before)
    return m, scaler, moved


def test_fp16_scaler_scales_and_updates_params():
    # Under amp_dtype='fp16' with an ENABLED scaler, a train_step must move parameters
    # (i.e. fp16 grads are scaled, not underflowed to zero).
    cfg = _tiny_cfg(amp_dtype="fp16")
    m, scaler, moved = _run_one_fp16_step(cfg)
    assert torch.isfinite(torch.tensor(scaler.get_scale()))
    assert moved, "no parameter moved under the enabled fp16 scaler (gradients underflowed?)"


def test_fp16_with_pullback_group_steps_the_k10_two_gl5_frame(monkeypatch):
    cfg = _tiny_cfg(
        amp_dtype="fp16",
        embed_dim=10,
        n_heads=2,
        gauge_group="block_glk",
        m_phi_update_mode="pullback_group",
        transport_chart_max_norm=6.0,
        pos_phi="learned",
    )
    torch.manual_seed(0)
    m = VFEModel(cfg)
    opt = build_optimizer(m, cfg); sch = _sched(opt)
    scaler = torch.amp.GradScaler(device="cpu", enabled=True)
    phi0 = m.prior_bank.phi_embed.detach().clone()
    staged_dtypes = []
    original_stage = gauge_optim.stage_pullback_group_candidate

    def _stage_spy(*args, **kwargs):
        candidate = original_stage(*args, **kwargs)
        staged_dtypes.append(candidate.candidate_phi.dtype)
        return candidate

    monkeypatch.setattr(gauge_optim, "stage_pullback_group_candidate", _stage_spy)
    tok, tgt = _batch(cfg)
    train_step(m, opt, sch, tok, tgt, grad_clip=1.0, scaler=scaler)
    assert m.group.irrep_dims == [5, 5]
    assert staged_dtypes and set(staged_dtypes) == {torch.float64}
    assert m.prior_bank.phi_embed.dtype == torch.float32
    assert m.pos_phi_free.dtype == torch.float32
    assert not torch.equal(phi0, m.prior_bank.phi_embed), "gauge frame did not move under fp16 scaler"


def test_fp16_pullback_overflow_preserves_all_phi_factors_and_clock(monkeypatch):
    cfg = _tiny_cfg(
        amp_dtype="fp16",
        embed_dim=10,
        gauge_group="block_glk",
        m_phi_update_mode="pullback_group",
        n_heads=2,
        transport_chart_max_norm=6.0,
        pos_phi="none",
    )
    torch.manual_seed(0)
    model = VFEModel(cfg)
    assert model.group.irrep_dims == [5, 5]
    optimizer = build_optimizer(model, cfg)
    scheduler = _sched(optimizer)
    scaler = torch.amp.GradScaler(device="cpu", enabled=True, init_scale=128.0)
    model.prior_bank.output_proj_weight.register_hook(
        lambda gradient: torch.full_like(gradient, float("inf"))
    )
    factors = (
        model.prior_bank.phi_embed,
    )
    before = [parameter.detach().clone() for parameter in factors]
    scale_before = float(scaler.get_scale())
    scheduler_epoch = scheduler.last_epoch
    staging_calls = []
    original_stage = gauge_optim.stage_pullback_group_candidate

    def _stage_spy(*args, **kwargs):
        staging_calls.append(1)
        return original_stage(*args, **kwargs)

    monkeypatch.setattr(gauge_optim, "stage_pullback_group_candidate", _stage_spy)
    status = {}
    tok, tgt = _batch(cfg)
    train_step(
        model,
        optimizer,
        scheduler,
        tok,
        tgt,
        grad_clip=1.0,
        scaler=scaler,
        status_out=status,
    )

    assert status["did_step"] is False and status["nonfinite_skip"] is True
    assert float(scaler.get_scale()) < scale_before
    assert scheduler.last_epoch == scheduler_epoch
    assert staging_calls == []
    for parameter, expected in zip(factors, before):
        assert torch.equal(parameter, expected)

