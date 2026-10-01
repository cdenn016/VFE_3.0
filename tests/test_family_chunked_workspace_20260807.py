r"""``family_chunked`` decode: bound the FULL-family functional workspace, and gate the checkpoint
on what not-checkpointing actually retains (audit 2026-08-07).

Two defects, both reachable only when ``decode_mode='family_chunked'`` meets a full family, and both
measured on an RTX 5090 before the fix.

**(1) One ``decode_chunk_size`` knob, two workspace units.** Five decode kernels read
``decode_chunk_size`` (prior_bank.py:1185, 1382, 1496, 1580, 1673) and it is sized for the ones whose
per-slice workspace is ``(B, N, Vc)``. A full family's is ``(B, N, Vc, K, K)`` -- ``K*K`` times larger
-- and nothing scaled the value on the way in: ``config.py:2515`` validates only ``>= 1``, and
``model.py:1609`` passes no ``chunk_size`` override, so the raw integer reached the loop. At the live
``B=64, N=128, K=20, V=50257, decode_chunk_size=8192`` that is a measured
``RuntimeError: CUDA out of memory. Tried to allocate 100.00 GiB`` on the FIRST slice, against a
6887 MiB peak for the whole ``full_chunked`` decode at the identical shape. The
``train_vfe3.py:161`` comment ("set chunks to 512 or default/K^2") was an unenforced suggestion, so
the failure mode was a hand-tuning obligation the config never stated.

**(2) The checkpoint gate measured a quantity the knob cannot move.** ``_decode_ce_should_checkpoint``
was handed ONE slice's bytes, but declining to checkpoint retains EVERY slice's workspace into
backward at once. That total is ``B*N*V*inner*itemsize`` -- invariant under the slice width. So
narrowing the slice to bound the transient (defect 1's fix) would have walked the per-slice estimate
under the 2 GiB threshold and silently switched checkpointing OFF, retaining the whole vocabulary
instead of one slice: strictly worse than the OOM it was fixing. The estimate was also 2x low on its
own terms, counting one ``(B, N, Vc, K, K)`` tensor where ``_full_gaussian_kl_terms`` allocates two
simultaneously (the forward and back substitutions at families/gaussian.py:117-118).

Pins: (a) the ceiling binds for a full family and is inert for a diagonal one; (b) narrowing is
value- and gradient-identical, because slicing a logsumexp is a tiling choice; (c) the checkpoint
decision is invariant under ``decode_chunk_size``; (d) the workspace estimate counts both solve
buffers; (e) the slice width never reaches zero, however small the budget.
"""

from tests.profiles import test_config
import warnings

import pytest
import torch

import vfe3.model.prior_bank as prior_bank
from vfe3.families.gaussian import FullGaussian, set_full_cov_kl_precision
from vfe3.model.prior_bank import (
    DECODE_CE_FAMILY_WORKSETS,
    _decode_ce_chunk_activation_bytes,
    _decode_ce_family_effective_chunk,
    _decode_ce_should_checkpoint,
)

from tests.test_amp import _tiny_model


def _bank(*, renyi_order=0.5, **kw):
    model = _tiny_model(gauge_group="block_glk", n_heads=2, family="gaussian_full",
                        decode_mode="family_chunked", renyi_order=renyi_order, **kw)
    return model, model.prior_bank


def _decode_mu_grad(pb):
    r"""Gradient of the (V, K) mean table the decode actually scores against.

    ``_decode_mu_table()`` is ``decode_mu_embed`` under the default ``untie_decode_bank=True`` and
    ``mu_embed`` on a tied bank; the fused CE on precomputed (mu, sigma) reaches only that table.
    """
    return pb._decode_mu_table().grad


def _non_default_decode_av_precision():
    r"""A ``decode_av_precision`` value that differs from the dataclass default.

    Inert-setting notices display only non-default values; activity itself is the structural
    ``inactive_fields()`` map, which does not depend on the value.
    """
    from vfe3.config import VFE3Config

    default = VFE3Config.__dataclass_fields__["decode_av_precision"].default
    return "fp32" if default != "fp32" else "fp64"


def _inputs(model):
    B, N, K = 2, 5, model.cfg.embed_dim
    g = torch.Generator().manual_seed(0)
    mu = torch.randn(B, N, K, generator=g)
    sigma = torch.eye(K).expand(B, N, K, K).clone()
    return mu, sigma, torch.randint(0, 20, (B, N), generator=g)


# -- (a) the ceiling binds for a full family, and is inert for a diagonal one -------------------

def test_effective_chunk_narrows_a_full_family_to_the_workspace_ceiling():
    r"""At the live shape the raw 8192 is cut to whatever keeps two (B,N,Vc,K,K) buffers under 1 GiB."""
    ref = torch.empty(64, 128, 20)                       # live B, N; fp32
    inner = 20 * 20
    got = _decode_ce_family_effective_chunk(ref, 8192, inner)

    per_entry = 64 * 128 * inner * DECODE_CE_FAMILY_WORKSETS * 4
    assert got == prior_bank.DECODE_CE_FAMILY_WORKSPACE_BYTES // per_entry
    assert got < 8192, "the ceiling did not bind at the shape that measured a 100 GiB allocation"
    # The bound it advertises actually holds.
    assert got * per_entry <= prior_bank.DECODE_CE_FAMILY_WORKSPACE_BYTES


def test_effective_chunk_is_inert_for_a_diagonal_family():
    r"""inner == 1 is every (B, N, Vc) kernel: the knob keeps its exact meaning, byte-identical."""
    ref = torch.empty(64, 128, 20)
    assert _decode_ce_family_effective_chunk(ref, 8192, 1) == 8192
    assert _decode_ce_family_effective_chunk(ref, 3, 1) == 3


def test_effective_chunk_never_returns_zero():
    r"""A budget smaller than one vocabulary entry must still make progress, not hang on range(0,V,0)."""
    ref = torch.empty(4096, 4096, 8)
    assert _decode_ce_family_effective_chunk(ref, 8192, 64 * 64) >= 1


def test_effective_chunk_never_widens_the_request():
    r"""The ceiling is a cap, not a target: a user asking for a narrow slice keeps it."""
    ref = torch.empty(2, 5, 20)
    assert _decode_ce_family_effective_chunk(ref, 3, 20 * 20) == 3


def test_effective_chunk_honors_an_explicit_fp64_scalar_cost(monkeypatch):
    r"""An fp64 functional workspace admits half as many entries as fp32 at the same budget."""
    monkeypatch.setattr(prior_bank, "DECODE_CE_FAMILY_WORKSPACE_BYTES", 960)
    ref = torch.empty(1, 1, 4)

    fp32_width = _decode_ce_family_effective_chunk(
        ref, 100, inner=10, workspace_bytes_per_scalar=4,
    )
    fp64_width = _decode_ce_family_effective_chunk(
        ref, 100, inner=10, workspace_bytes_per_scalar=8,
    )

    assert fp32_width == 12
    assert fp64_width == 6


@pytest.mark.parametrize("invalid", [0, -1, 1.5, True])
def test_workspace_scalar_byte_overrides_require_positive_plain_integers(invalid):
    r"""Byte counts are discrete; floats and bools must not leak into workspace arithmetic."""
    ref = torch.empty(1, 1, 4)

    with pytest.raises(ValueError, match="positive integer"):
        _decode_ce_chunk_activation_bytes(ref, 1, workspace_bytes_per_scalar=invalid)
    with pytest.raises(ValueError, match="positive integer"):
        _decode_ce_family_effective_chunk(
            ref, 1, inner=4, workspace_bytes_per_scalar=invalid,
        )


# -- (b) narrowing is value- and gradient-identical ---------------------------------------------

@pytest.fixture
def spy(monkeypatch):
    r"""Record the vocab width of every functional call the decode makes."""
    widths = []
    real = prior_bank.get_functional

    def _wrapped(name):
        fn = real(name)

        def _spy(q, p, **kw):
            out = fn(q, p, **kw)
            widths.append(out.shape[-1])
            return out
        return _spy

    monkeypatch.setattr(prior_bank, "get_functional", _wrapped)
    return widths


def _one_entry_budget(model):
    r"""Exactly one vocabulary entry's workspace at the ``_inputs`` shape -> maximal retiling.

    Derived from the model's real K rather than hard-coded: ``_tiny_model`` defaults to
    ``embed_dim=4``, and a budget written against a guessed K silently fails to bind, which makes
    every value-equality assertion below compare a tiling against itself.
    """
    B, N, K = 2, 5, model.cfg.embed_dim
    return B * N * K * K * DECODE_CE_FAMILY_WORKSETS * 4      # fp32


def test_narrowed_slices_match_the_unnarrowed_ce(monkeypatch, spy):
    r"""Forcing the ceiling to bind must not move the loss: slicing a logsumexp is a tiling choice."""
    model, pb = _bank(decode_chunk_size=20)
    mu, sigma, targets = _inputs(model)
    V = pb.vocab_size

    wide = pb.decode_ce_family_chunked(mu, sigma, targets)
    wide_widths = list(spy)
    spy.clear()

    # Squeeze the budget so the same call is forced into many narrow slices.
    monkeypatch.setattr(prior_bank, "DECODE_CE_FAMILY_WORKSPACE_BYTES", _one_entry_budget(model))
    narrow = pb.decode_ce_family_chunked(mu, sigma, targets)

    # The retiling MUST actually have happened, or the equality below proves nothing.
    assert max(narrow_w := spy) == 1 and len(narrow_w) == V, f"ceiling did not bind: {narrow_w}"
    assert len(wide_widths) < len(narrow_w), "the two runs used the same tiling"
    assert sum(wide_widths) == sum(narrow_w) == V, "a tiling dropped vocabulary entries"

    assert torch.allclose(wide, narrow, rtol=0, atol=1e-6), f"{wide.item()} != {narrow.item()}"


def test_narrowed_slices_match_the_unnarrowed_gradient(monkeypatch, spy):
    r"""The gradient to the prior tables must survive the retiling too, not just the value."""
    model, pb = _bank(decode_chunk_size=20)
    mu, sigma, targets = _inputs(model)

    def _grad():
        pb.zero_grad(set_to_none=True)
        pb.decode_ce_family_chunked(mu, sigma, targets).backward()
        return _decode_mu_grad(pb).clone()

    wide = _grad()
    n_wide = len(spy)
    spy.clear()

    monkeypatch.setattr(prior_bank, "DECODE_CE_FAMILY_WORKSPACE_BYTES", _one_entry_budget(model))
    narrow = _grad()

    assert len(spy) > n_wide, "the ceiling did not bind -- the comparison would be vacuous"
    assert torch.allclose(wide, narrow, rtol=0, atol=1e-6)
    assert torch.isfinite(narrow).all()
    assert narrow.abs().sum() > 0, "a zero gradient would satisfy the equality vacuously"


def test_registered_chunked_logits_tile_full_family_at_the_workspace_ceiling(monkeypatch, spy):
    r"""``decode`` must bound the real full-family functional, not merely fused CE."""
    model, pb = _bank(decode_chunk_size=20)
    mu, sigma, _ = _inputs(model)
    V = pb.vocab_size
    monkeypatch.setattr(prior_bank, "DECODE_CE_FAMILY_WORKSPACE_BYTES", _one_entry_budget(model))

    logits = pb.decode(mu, sigma)

    assert logits.shape[-1] == V
    assert spy == [1] * V


def test_registered_chunked_logits_preserve_values_when_workspace_retiles(monkeypatch, spy):
    r"""Changing only the workspace tiling cannot change logits or omit a vocabulary slice."""
    model, pb = _bank(decode_chunk_size=20)
    mu, sigma, _ = _inputs(model)
    V = pb.vocab_size

    wide = pb.decode(mu, sigma)
    wide_widths = list(spy)
    spy.clear()
    monkeypatch.setattr(prior_bank, "DECODE_CE_FAMILY_WORKSPACE_BYTES", _one_entry_budget(model))
    narrow = pb.decode(mu, sigma)

    assert wide_widths == [V]
    assert spy == [1] * V
    assert sum(spy) == V
    assert torch.allclose(wide, narrow, rtol=0, atol=1e-6)


def test_dense_family_logits_remain_one_full_vocabulary_call_under_workspace_cap(monkeypatch, spy):
    r"""The dense ``family`` mode intentionally has no workspace-retiling contract."""
    model = _tiny_model(
        gauge_group="block_glk", n_heads=2, family="gaussian_full", decode_mode="family",
        renyi_order=0.5, decode_chunk_size=20, _allow_legacy_dense_decode=True,
    )
    pb = model.prior_bank
    mu, sigma, _ = _inputs(model)
    monkeypatch.setattr(prior_bank, "DECODE_CE_FAMILY_WORKSPACE_BYTES", _one_entry_budget(model))

    logits = pb.decode(mu, sigma)

    assert logits.shape[-1] == pb.vocab_size
    assert spy == [pb.vocab_size]


def test_fp64_full_family_uses_eight_byte_workspace_for_fused_ce_and_registered_logits(
        monkeypatch, spy):
    r"""The full Gaussian's fp64 compute island governs both public fp32 family routes."""
    previous = set_full_cov_kl_precision("fp64")
    try:
        model, pb = _bank(decode_chunk_size=20)
        mu, sigma, targets = _inputs(model)
        V = pb.vocab_size
        # This admits all V fp32 entries but only half as many fp64 entries.
        fp32_whole_vocab = _one_entry_budget(model) * V
        monkeypatch.setattr(prior_bank, "DECODE_CE_FAMILY_WORKSPACE_BYTES", fp32_whole_vocab)

        ce = pb.decode_ce_family_chunked(mu, sigma, targets)
        fused_widths = list(spy)
        spy.clear()
        logits = pb.decode(mu, sigma)

        assert torch.isfinite(ce)
        assert logits.shape[-1] == V
        assert fused_widths == [V // 2, V - V // 2]
        assert spy == [V // 2, V - V // 2]
    finally:
        set_full_cov_kl_precision(previous)


def test_fp64_full_family_checkpoint_gate_uses_eight_byte_whole_vocabulary_estimate(monkeypatch):
    r"""The fp64 full family must checkpoint when its eight-byte estimate crosses the threshold."""
    previous = set_full_cov_kl_precision("fp64")
    try:
        model, pb = _bank(decode_chunk_size=20, decode_ce_checkpoint="auto")
        mu, sigma, targets = _inputs(model)
        V = pb.vocab_size
        four_byte = _decode_ce_chunk_activation_bytes(
            mu, V, inner=model.cfg.embed_dim ** 2 * DECODE_CE_FAMILY_WORKSETS,
        )
        eight_byte = 2 * four_byte
        assert four_byte < 38_400 < eight_byte
        monkeypatch.setattr(prior_bank, "DECODE_CE_CHECKPOINT_AUTO_BYTES", 38_400)
        seen = []
        real_gate = prior_bank._decode_ce_should_checkpoint

        def _recording_gate(mode, grad_active, activation_bytes):
            decision = real_gate(mode, grad_active, activation_bytes)
            seen.append((activation_bytes, decision))
            return decision

        monkeypatch.setattr(prior_bank, "_decode_ce_should_checkpoint", _recording_gate)
        pb.decode_ce_family_chunked(mu, sigma, targets)

        assert seen == [(eight_byte, True)]
    finally:
        set_full_cov_kl_precision(previous)


def test_fp64_workspace_override_leaves_diagonal_family_sizing_unchanged():
    r"""Only full-family workspaces enter the fp64 island; diagonal family sizing stays inert."""
    previous = set_full_cov_kl_precision("fp64")
    try:
        ref = torch.empty(2, 5, 4)
        assert _decode_ce_family_effective_chunk(
            ref, 17, inner=1, workspace_bytes_per_scalar=8,
        ) == 17
    finally:
        set_full_cov_kl_precision(previous)


@pytest.mark.parametrize("policy", ["fp32_escalate", "fp32_escalate_cond"])
def test_full_gaussian_workspace_reserves_fp64_for_escalation_policies(policy):
    r"""Both escalation policies can recompute a full-family grid in fp64."""
    previous = set_full_cov_kl_precision(policy)
    try:
        ref = torch.empty(2, 5, 4)
        assert prior_bank._full_family_workspace_bytes_per_scalar(FullGaussian, ref, ref) == 8
    finally:
        set_full_cov_kl_precision(previous)


def test_declining_the_prior_hoist_still_promotes_each_slice(monkeypatch, spy):
    r"""The (V, K, K) promotion is hoisted only when it is small; the fallback must still promote.

    A full family needs a ``(Vc, K, K)`` prior slice. When the whole-table hoist is declined (the
    table would be the new memory problem -- 8.9 GB at K=210) the loop must ``diag_embed`` each
    slice itself. Slicing the UN-promoted ``(V, K)`` table instead hands the functional a diagonal
    dispersion for a full family, which is a shape error at best and a silently wrong divergence at
    worst -- so this pins the fallback against the hoisted path, not merely against "it ran".
    """
    model, pb = _bank(decode_chunk_size=20)
    mu, sigma, targets = _inputs(model)

    hoisted = pb.decode_ce_family_chunked(mu, sigma, targets)
    spy.clear()

    # One byte under the promoted table's size: the hoist is declined, the loop must promote.
    V, K = pb.vocab_size, model.cfg.embed_dim
    monkeypatch.setattr(prior_bank, "DECODE_CE_FAMILY_WORKSPACE_BYTES", V * K * K * 4 - 1)
    fallback = pb.decode_ce_family_chunked(mu, sigma, targets)

    assert spy, "the decode made no functional call"
    assert torch.allclose(hoisted, fallback, rtol=0, atol=1e-6)
    assert torch.isfinite(fallback)


# -- (c) the checkpoint decision does not depend on the slice width -----------------------------

def test_checkpoint_gate_is_invariant_under_chunk_size():
    r"""Not-checkpointing retains every slice, so the decision is keyed on B*N*V, not B*N*Vc.

    This is the regression that made defect 1's fix dangerous: under the old per-slice estimate,
    narrowing the slice walked the number under the threshold and turned checkpointing off.
    """
    ref = torch.empty(64, 128, 20)
    V, inner = 50257, 20 * 20 * DECODE_CE_FAMILY_WORKSETS

    total = _decode_ce_chunk_activation_bytes(ref, V, inner=inner)
    decisions = {
        _decode_ce_should_checkpoint("auto", True, total)          # gate sees the SAME total
        for _ in (1, 20, 512, 8192)                                 # regardless of slice width
    }
    assert decisions == {True}, "the whole-vocabulary retention must still trip the auto threshold"

    # And the old per-slice estimate really would have flipped it -- this is not a hypothetical.
    per_slice = _decode_ce_chunk_activation_bytes(ref, 40, inner=inner)
    assert not _decode_ce_should_checkpoint("auto", True, per_slice)


# -- (d) the estimate counts BOTH triangular-solve buffers --------------------------------------

def test_workspace_estimate_counts_both_solve_buffers():
    r"""_full_gaussian_kl_terms holds Y and Z simultaneously; counting one halved the figure."""
    assert DECODE_CE_FAMILY_WORKSETS == 2
    ref = torch.empty(8, 32, 20)
    one = _decode_ce_chunk_activation_bytes(ref, 512, inner=400)
    both = _decode_ce_chunk_activation_bytes(ref, 512, inner=400 * DECODE_CE_FAMILY_WORKSETS)
    assert both == 2 * one


# -- (f) decode_av_precision is inert only on generic family routes -------------------------------

def test_decode_av_precision_is_reported_inert_on_the_family_route():
    r"""The canonical full family dispatch reads the precision policy; generic routes do not.

    Under built-in full Gaussian / built-in Renyi(alpha=1) with ``full_cov_kl_precision='fp32_escalate'``,
    switching ``decode_av_precision`` selects between the analytic and generic family routes, so it
    must not be reported inert. A noncanonical order remains generic and must retain the warning.
    """
    from vfe3.config import VFE3Config

    base = dict(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=1,
                family="gaussian_full", gauge_group="block_glk", lambda_alpha_mode="constant")
    non_default = _non_default_decode_av_precision()

    def _inert_mentions(mode, **overrides):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            cfg = test_config(**base, decode_mode=mode, decode_av_precision=non_default, **overrides)
        shown = any("decode_av_precision" in str(r.message) for r in record)
        structural = "decode_av_precision" in cfg.inactive_fields()
        assert shown == structural, f"{mode}: notice {shown} != inactive_fields() {structural}"
        return structural

    canonical = dict(renyi_order=1.0, full_cov_kl_precision="fp32_escalate")
    assert not _inert_mentions("family_chunked", **canonical)
    assert _inert_mentions("family_chunked", renyi_order=0.5,
                           full_cov_kl_precision="fp32_escalate")
    # full_chunked genuinely reads it, so it must NOT be reported inert there.
    assert not _inert_mentions("full_chunked"), "full_chunked reads _decode_av -- not inert"


def test_same_name_family_chunked_override_keeps_decode_av_precision_inert():
    r"""The mode name alone must not make a custom family decoder inherit the analytic dependency."""
    from vfe3.config import VFE3Config

    original = prior_bank.get_decode_registration("family_chunked")

    def generic_logits(pb, mu_q, sigma_q, tau_eff):
        return prior_bank._family_logits(pb, mu_q, sigma_q, tau_eff, chunk=pb.decode_chunk_size)

    def generic_fused_ce(pb, mu_q, sigma_q, targets, *, tau=None,
                         chunk_size=None, ignore_index=-100):
        logits = generic_logits(pb, mu_q, sigma_q, pb._tau_eff(tau))
        valid = targets != ignore_index
        local_targets = targets.clamp_min(0)
        ce = (torch.logsumexp(logits, dim=-1) - logits.gather(
            -1, local_targets.unsqueeze(-1),
        ).squeeze(-1))
        return (ce * valid).sum() / valid.sum().clamp_min(1)

    prior_bank.register_decode(
        "family_chunked",
        covariance_kinds=frozenset({"diagonal", "full"}),
        family_consistent=True,
        supports_chunked=True,
        fused_ce=generic_fused_ce,
        can_omit_base_mean=True,
        can_omit_base_variance=True,
        override=True,
    )(generic_logits)
    try:
        with pytest.warns(Warning) as record:
            cfg = test_config(
                vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=1,
                family="gaussian_full", gauge_group="block_glk",
                decode_mode="family_chunked", renyi_order=1.0,
                full_cov_kl_precision="fp32_escalate",
                decode_av_precision=_non_default_decode_av_precision(),
            )
        assert "decode_av_precision" in cfg.inactive_fields()
        assert any("decode_av_precision" in str(warning.message) for warning in record)
    finally:
        prior_bank.register_decode(
            "family_chunked",
            supports_full=original.supports_full,
            supports_chunked=original.supports_chunked,
            family_consistent=original.family_consistent,
            fused_ce=original.fused_ce,
            covariance_kinds=original.covariance_kinds,
                can_omit_base_mean=original.can_omit_base_mean,
                can_omit_base_variance=original.can_omit_base_variance,
                fused_ce_supports_stats=original.fused_ce_supports_stats,
                override=True,
        )(original.callable)

    restored = prior_bank.get_decode_registration("family_chunked")
    assert restored.callable is original.callable
    assert restored.fused_ce is original.fused_ce
    assert restored == original


# -- (e) the route still runs end to end at a binding ceiling ------------------------------------

@pytest.mark.parametrize("multiple", [1, 8])
def test_forward_backward_is_finite_at_a_binding_ceiling(monkeypatch, multiple):
    model, pb = _bank(decode_chunk_size=8192)
    monkeypatch.setattr(prior_bank, "DECODE_CE_FAMILY_WORKSPACE_BYTES",
                        multiple * _one_entry_budget(model))
    mu, sigma, targets = _inputs(model)

    ce = pb.decode_ce_family_chunked(mu, sigma, targets)
    ce.backward()

    assert torch.isfinite(ce)
    assert _decode_mu_grad(pb) is not None and torch.isfinite(_decode_mu_grad(pb)).all()
