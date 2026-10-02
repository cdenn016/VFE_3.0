r"""Opt-in mixed-precision (amp_dtype) tests.

The pure fp32 path (amp_dtype=None, the default) must not enable autocast and must stay
byte-identical to the no-AMP build; the bf16/fp16 path must run and stay finite; and the
cancellation-sensitive decode + cross-entropy must stay fp32 under an explicitly selected fp32 policy even when AMP is on (the fp32
island holds). The full suite is the broader default-off byte-identity oracle (every other
test runs amp_dtype=None); these tests pin the AMP wiring specifically.
"""

from tests.profiles import test_config
from contextlib import nullcontext
from unittest import mock

import numpy as np
import torch

from vfe3.model.model import VFEModel
from vfe3.train import _val_diagnostics
from vfe3.belief import BeliefState
from vfe3.viz import figures as figs
from vfe3.viz.extract import _cpu_bank_value


def _tiny_model(**overrides) -> VFEModel:
    base = dict(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=1,
                n_e_steps=2, e_q_mu_lr=0.05,
                decode_mode="diagonal_chunked", decode_av_precision="fp32",
                decode_ranking_fp64_escalation=False, lambda_alpha_mode="constant",
                mu_init_std=0.02, sigma_init=1.0, phi_scale=0.01)
    base.update(overrides)
    return VFEModel(test_config(**base))


def test_default_off_enters_no_autocast_context():
    # Disabling autocast for numerical islands is safe; enabling AMP is not.
    model = _tiny_model()
    assert model.cfg.amp_dtype is None
    tok = torch.randint(0, 20, (2, 5)); tgt = torch.randint(0, 20, (2, 5))

    real_autocast = torch.autocast

    def _tripwire(*a, **k):
        assert k.get("enabled", True) is False
        return real_autocast(*a, **k)

    with mock.patch("torch.autocast", _tripwire):
        _, loss, _ = model(tok, tgt)
    assert torch.isfinite(loss)


def test_default_off_forward_is_deterministic_and_finite():
    # The default (amp_dtype=None) forward is unchanged: same seed -> identical logits/loss.
    tok = torch.randint(0, 20, (2, 5)); tgt = torch.randint(0, 20, (2, 5))
    torch.manual_seed(0); m1 = _tiny_model()
    torch.manual_seed(0); m2 = _tiny_model()
    _, loss1, _ = m1(tok, tgt)
    _, loss2, _ = m2(tok, tgt)
    # logits is None on the fused-CE branch by design (audit 2026-08-06) -- the default
    # decode_mode is chunked -- so compare logits via the unfused no-targets path.
    assert torch.equal(m1(tok), m2(tok)) and torch.equal(loss1, loss2)


def test_bf16_forward_runs_and_is_finite():
    # amp_dtype='bf16' wraps the E-step in autocast (device-resolved, so the CPU box exercises
    # it) and returns a finite loss. CPU bf16 autocast is REAL casting (not a no-op), so the loss
    # is only loosely close to fp32 -- assert finite + shape + a relaxed allclose. The point is the
    # path is wired and the fp32 islands hold (no crash, finite).
    tok = torch.randint(0, 20, (2, 5)); tgt = torch.randint(0, 20, (2, 5))
    torch.manual_seed(0); m_fp32 = _tiny_model()
    torch.manual_seed(0); m_bf16 = _tiny_model(amp_dtype="bf16")
    _, loss_fp32, _ = m_fp32(tok, tgt)
    _, loss_bf16, _ = m_bf16(tok, tgt)
    assert m_bf16(tok).shape == m_fp32(tok).shape      # unfused path; fused CE returns no logits
    assert torch.isfinite(loss_bf16)
    # bf16 mantissa is ~3 decimal digits; the loss should be in the same ballpark as fp32.
    assert torch.allclose(loss_bf16, loss_fp32, atol=0.5, rtol=0.1)


def test_fp16_forward_runs_and_is_finite():
    # amp_dtype='fp16' is accepted and runs a forward without error (forward-correctness scope;
    # fp16 TRAINING would need a GradScaler in the M-step -- a documented follow-up).
    tok = torch.randint(0, 20, (2, 5)); tgt = torch.randint(0, 20, (2, 5))
    m = _tiny_model(amp_dtype="fp16")
    _, loss, _ = m(tok, tgt)
    assert torch.isfinite(loss)


def test_decode_and_ce_stay_fp32_under_bf16():
    # The cancellation-sensitive decode + CE must run in fp32 even under amp_dtype='bf16': the
    # island holds. Intercept PriorBank.decode to record its INPUT dtype (the .float() guard) and
    # its OUTPUT logits dtype, and assert the CE-input logits are fp32. This pins the protection
    # the islands exist for (the atol-1e-3 decode pin) regardless of the autocast E-step.
    tok = torch.randint(0, 20, (2, 5)); tgt = torch.randint(0, 20, (2, 5))
    model = _tiny_model(amp_dtype="bf16")

    seen = {}
    real_decode = model.prior_bank.decode

    def _spy(mu_q, sigma_q, **kw):
        seen["mu_in"] = mu_q.dtype
        seen["sigma_in"] = sigma_q.dtype
        out = real_decode(mu_q, sigma_q, **kw)
        seen["logits_out"] = out.dtype
        return out

    # No targets: the fused-CE branch bypasses prior_bank.decode entirely (it calls the registered
    # fused_ce), so the spy would never fire and this dtype contract would go unchecked. The
    # unfused path is the one that runs `decode`, which is what this test is about (audit
    # 2026-08-06). The fused branch's own finiteness is asserted separately below.
    with mock.patch.object(model.prior_bank, "decode", _spy):
        model(tok)
    _, loss, _ = model(tok, tgt)

    # Decode inputs are .float()-ed (the load-bearing fp32 guard), so the cancellation-sensitive
    # matmul runs in fp32, and the logits feeding CE are fp32.
    assert seen["mu_in"] == torch.float32
    assert seen["sigma_in"] == torch.float32
    assert seen["logits_out"] == torch.float32
    assert torch.isfinite(loss)


def test_bf16_backward_reaches_prior_tables():
    # Under bf16 the loss must still backprop through the unrolled E-step to the prior tables
    # (the AMP wrap does not sever the training graph). bf16 needs no GradScaler.
    tok = torch.randint(0, 20, (2, 5)); tgt = torch.randint(0, 20, (2, 5))
    model = _tiny_model(amp_dtype="bf16")
    _, loss, _ = model(tok, tgt)
    loss.backward()
    assert model.prior_bank.mu_embed.grad is not None
    assert model.prior_bank.mu_embed.grad.abs().sum() > 0


def test_bf16_head_mixer_validation_diagnostics_run_in_fp32():
    r"""A bf16 head-mixer forward must leave the off-graph diagnostic replay in fp32."""
    model = _tiny_model(
        amp_dtype="bf16",
        gauge_group="block_glk",
        use_head_mixer=True,
    )
    tok = torch.randint(0, 20, (2, 5))
    tgt = torch.randint(0, 20, (2, 5))

    model(tok, tgt)  # match the successful validation forward that precedes the diagnostic replay
    snapshot = model.build_diagnostic_snapshot(tok)
    diagnostics = model.diagnostics(tok, snapshot=snapshot)
    val_diagnostics = _val_diagnostics(model, [(tok, tgt)], torch.device("cpu"))

    assert snapshot.stack_output.mu.dtype == torch.float32
    assert snapshot.stack_output.sigma.dtype == torch.float32
    assert snapshot.beta_maps.dtype == torch.float32
    assert diagnostics["relational_legacy_global_f_available"] == 0.0
    assert torch.isfinite(torch.tensor(diagnostics["relational_receiver_cost_after"]))
    assert torch.isfinite(torch.tensor(val_diagnostics.metrics["val_relational_receiver_cost_after"]))


# ---------------------------------------------------------------------------------------------
# The AMP -> analysis boundary: report banks and the numpy conversion behind every figure.
# Under amp_dtype='bf16' the autocast E-step returns bf16 mu/sigma (phi keeps its own fp32
# island), and NumPy has NO bfloat16 dtype -- .numpy() raises "Got unsupported ScalarType
# BFloat16". The diagnostic snapshot already upcasts at capture (_freeze_tensor, model.py);
# these pin the same invariant for the viz bank capture and the figure numpy boundary.
# ---------------------------------------------------------------------------------------------

def test_cpu_bank_value_upcasts_reduced_precision_to_fp32():
    # The bank-capture helper is the viz sibling of model._freeze_tensor: it must upcast the
    # autocast forward's bf16/fp16 beliefs to fp32 so downstream analysis (numpy, eigh,
    # sklearn) receives a representable dtype. Integer and fp32 fields must pass through
    # untouched, so the amp_dtype=None default path stays byte-identical.
    for reduced in (torch.bfloat16, torch.float16):
        hosted = _cpu_bank_value(torch.ones(3, dtype=reduced))
        assert hosted.dtype == torch.float32, reduced

    ids = torch.arange(3)
    assert _cpu_bank_value(ids).dtype == torch.int64
    assert _cpu_bank_value(torch.ones(3)).dtype == torch.float32

    belief = BeliefState(
        mu=torch.ones(2, 3, dtype=torch.bfloat16),
        sigma=torch.ones(2, 3, dtype=torch.bfloat16),
        phi=torch.ones(2, 3),
    )
    hosted_belief = _cpu_bank_value(belief)
    assert hosted_belief.mu.dtype == torch.float32
    assert hosted_belief.sigma.dtype == torch.float32
    assert hosted_belief.phi.dtype == torch.float32


def test_np_conversion_accepts_bfloat16():
    # Every figure funnels its tensors through figures._np. NumPy cannot represent bfloat16 at
    # all, so the helper must widen it; bf16 -> fp32 is exact (bf16 is a truncated fp32), so the
    # values must survive the widening unchanged.
    values = torch.tensor([-1.5, 0.0, 2.25], dtype=torch.bfloat16)
    converted = figs._np(values)
    assert converted.dtype == np.float32
    assert np.array_equal(converted, np.array([-1.5, 0.0, 2.25], dtype=np.float32))


def test_sigma_calibration_figures_render_from_a_bf16_bank(tmp_path):
    # The two figures the bf16 run lost outright: both read belief_ce_bank['tr_sigma'], which
    # carries the autocast E-step's dtype straight from metrics.sigma_trace(out.sigma).
    generator = torch.Generator().manual_seed(0)
    bank = {
        "tr_sigma": torch.rand(64, generator=generator).to(torch.bfloat16),
        "ce":       torch.rand(64, generator=generator).to(torch.bfloat16),
    }
    stratified = tmp_path / "sigma_stratified_error.png"
    scatter = tmp_path / "sigma_ce_scatter.png"
    figures = [figs.plot_sigma_stratified_error(bank, n_bins=4, n_boot=8, path=str(stratified)),
               figs.plot_sigma_ce_scatter(bank, path=str(scatter))]
    for figure in figures:
        figs.plt.close(figure)                   # an open figure fails later tests' leak checks
    assert stratified.exists() and scatter.exists()
