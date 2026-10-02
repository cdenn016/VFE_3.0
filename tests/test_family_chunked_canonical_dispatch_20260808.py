r"""Canonical full-Gaussian ``family_chunked`` dispatch (2026-08-08).

The generic family decoder is the compatibility oracle: registrations, non-KL Renyi orders,
mixed public dtypes, and noncanonical precision policies must retain it.  Only the built-in
full-Gaussian / built-in Renyi(alpha=1) fp32 route may take the diagonal-prior analytic kernel.
"""

import os
from types import SimpleNamespace

import pytest
import torch

import vfe3.model.prior_bank as prior_bank
from tests.test_amp import _tiny_model
from vfe3.divergence import (
    get_family,
    get_functional,
    register_family,
    register_functional,
    renyi,
)
from vfe3.families.gaussian import (
    FullGaussian,
    full_cov_kl_precision,
    set_full_cov_kl_precision,
)


DEVICE = torch.device(os.environ.get("VFE3_TEST_DEVICE", "cpu"))


@pytest.fixture(autouse=True)
def _restore_process_globals():
    previous_full = set_full_cov_kl_precision("fp32_escalate")
    previous_av = prior_bank.set_decode_av_precision("fp32")
    previous_family = get_family("gaussian_full")
    previous_renyi = get_functional("renyi")
    try:
        yield
    finally:
        register_family("gaussian_full", override=True)(previous_family)
        register_functional("renyi", override=True)(previous_renyi)
        set_full_cov_kl_precision(previous_full)
        prior_bank.set_decode_av_precision(previous_av)


def _bank(*, decode_mode="family_chunked", renyi_order=1.0, divergence_family="renyi",
          decode_unigram_prior=True, untie_decode_bank=True):
    if renyi_order != 1.0 or divergence_family != "renyi":
        pb = prior_bank.PriorBank(
            vocab_size=17, K=4, n_gen=8, family="gaussian_full", diagonal_covariance=False,
            decode_mode=decode_mode, decode_chunk_size=5, renyi_order=renyi_order,
            divergence_family=divergence_family, decode_av_precision="fp32",
            untie_decode_bank=untie_decode_bank, decode_unigram_prior=decode_unigram_prior,
            unigram_kappa=0.7 if decode_unigram_prior else 1.0,
            full_cov_precision_policy="fp32_escalate",
        ).to(DEVICE)
        model = SimpleNamespace(cfg=SimpleNamespace(embed_dim=4), prior_bank=pb)
    else:
        model = _tiny_model(
            vocab_size=17,
            embed_dim=4,
            n_heads=2,
            max_seq_len=3,
            n_layers=1,
            gauge_group="block_glk",
            family="gaussian_full",
            full_cov_kl_precision="fp32_escalate",
            decode_mode=decode_mode,
            decode_chunk_size=5,
            untie_decode_bank=untie_decode_bank,
            decode_unigram_prior=decode_unigram_prior,
            unigram_kappa=0.7 if decode_unigram_prior else 1.0,
        ).to(DEVICE)
        pb = model.prior_bank
    # The dispatch contract deliberately targets the model-owned fp32 analytic configuration.
    set_full_cov_kl_precision("fp32_escalate")
    prior_bank.set_decode_av_precision("fp32")
    if decode_unigram_prior:
        pb.set_unigram_log_prior(torch.arange(1, pb.vocab_size + 1, device=DEVICE))
    return model, pb


def _spd_inputs():
    g = torch.Generator(device=DEVICE).manual_seed(20260808)
    B, N, K = 2, 3, 4
    mu = torch.randn(B, N, K, generator=g, device=DEVICE)
    lower = 0.15 * torch.randn(B, N, K, K, generator=g, device=DEVICE).tril(-1)
    factor = torch.eye(K, device=DEVICE).expand(B, N, K, K).clone() + lower
    factor.diagonal(dim1=-2, dim2=-1).add_(0.4)
    sigma = factor @ factor.transpose(-1, -2)
    targets = torch.tensor([[1, -100, 16], [4, 9, 3]], device=DEVICE)
    return mu, sigma, targets


def _canonical(pb):
    return (
        pb.full_cov_kl_precision == "fp32_escalate"
        and prior_bank.decode_av_precision() == "fp32"
        and pb.renyi_order == 1.0
    )


def _loss_and_grads(pb, mu, sigma, targets):
    pb.zero_grad(set_to_none=True)
    mu = mu.detach().clone().requires_grad_(True)
    sigma = sigma.detach().clone().requires_grad_(True)
    loss = pb.decode_ce_family_chunked(mu, sigma, targets, tau=0.73, chunk_size=5)
    loss.backward()
    return loss.detach(), [
        mu.grad.detach().clone(),
        sigma.grad.detach().clone(),
        pb.decode_mu_embed.grad.detach().clone(),
        pb.decode_sigma_log_embed.grad.detach().clone(),
        pb.decode_log_scale.grad.detach().clone(),
    ]


def test_canonical_fused_ce_and_all_decode_gradients_are_exactly_full_chunked():
    """A generic pair-grid dispatch would change this exact analytic reduction and its gradients."""
    _, pb = _bank()
    assert _canonical(pb)
    mu, sigma, targets = _spd_inputs()

    family_loss, family_grads = _loss_and_grads(pb, mu, sigma, targets)
    pb.zero_grad(set_to_none=True)
    mu_full = mu.detach().clone().requires_grad_(True)
    sigma_full = sigma.detach().clone().requires_grad_(True)
    full_loss = pb.decode_ce_full_chunked(
        mu_full, sigma_full, targets, tau=0.73, chunk_size=5)
    full_loss.backward()
    full_grads = [
        mu_full.grad.detach().clone(),
        sigma_full.grad.detach().clone(),
        pb.decode_mu_embed.grad.detach().clone(),
        pb.decode_sigma_log_embed.grad.detach().clone(),
        pb.decode_log_scale.grad.detach().clone(),
    ]

    assert torch.equal(family_loss, full_loss.detach())
    for got, want in zip(family_grads, full_grads):
        assert torch.equal(got, want)
        assert torch.isfinite(got).all() and got.abs().sum() > 0


def test_canonical_registered_logits_match_full_once_biased_and_backward():
    """Changing dispatch, or adding the unigram bias inside the delegate, fails this contract."""
    _, family_pb = _bank(decode_mode="family_chunked")
    _, full_pb = _bank(decode_mode="full_chunked")
    full_pb.load_state_dict(family_pb.state_dict())
    mu, sigma, _ = _spd_inputs()
    mu = mu.requires_grad_(True)

    family_logits = family_pb.decode(mu, sigma, tau=0.91)
    full_logits = full_pb.decode(mu, sigma, tau=0.91)
    direct_plus_one_bias = prior_bank._decode_full_chunked(
        family_pb, mu, sigma, family_pb._tau_eff(0.91),
    ) + family_pb._unigram_bias()
    assert torch.equal(family_logits, full_logits)
    assert torch.equal(family_logits, direct_plus_one_bias)

    weights = torch.linspace(-0.4, 0.8, family_logits.numel(), device=DEVICE).reshape_as(family_logits)
    (family_logits * weights).sum().backward()
    assert mu.grad is not None and torch.isfinite(mu.grad).all()
    assert family_pb.decode_mu_embed.grad is not None
    assert family_pb.decode_sigma_log_embed.grad is not None
    assert family_pb.decode_log_scale.grad is not None


def test_canonical_dispatch_has_no_pair_grid_solves_but_generic_renyi_does(monkeypatch):
    """Replacing the analytic delegate with the generic functional reintroduces triangular solves."""
    _, canonical_pb = _bank()
    _, analytic_pb = _bank(decode_mode="full_chunked")
    analytic_pb.load_state_dict(canonical_pb.state_dict())
    _, generic_pb = _bank(renyi_order=0.5)
    mu, sigma, _ = _spd_inputs()
    calls = 0
    real = torch.linalg.solve_triangular

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, "solve_triangular", counted)
    canonical_pb.decode(mu, sigma)
    canonical_calls = calls
    calls = 0
    analytic_pb.decode(mu, sigma)
    analytic_calls = calls
    calls = 0
    generic_pb.decode(mu, sigma)
    generic_calls = calls

    assert canonical_calls == analytic_calls == 0
    assert generic_calls > 0


def _assert_generic_oracle(pb, mu, sigma):
    got = pb.decode(mu, sigma, tau=0.83)
    want = pb.reference_decode(mu, sigma, tau=0.83)
    torch.testing.assert_close(got, want, atol=5e-6, rtol=2e-5)


@pytest.mark.parametrize("renyi_order,divergence_family", [(0.5, "renyi"), (1.0, "squared_hellinger")])
def test_noncanonical_family_settings_keep_the_generic_oracle(renyi_order, divergence_family):
    _, pb = _bank(renyi_order=renyi_order, divergence_family=divergence_family)
    mu, sigma, _ = _spd_inputs()
    _assert_generic_oracle(pb, mu, sigma)


def test_same_name_runtime_renyi_override_keeps_the_generic_oracle():
    _, pb = _bank()
    mu, sigma, _ = _spd_inputs()
    builtin = get_functional("renyi")
    calls = 0

    def overridden(*args, **kwargs):
        nonlocal calls
        calls += 1
        return builtin(*args, **kwargs)

    register_functional("renyi", override=True)(overridden)
    try:
        logits = pb.decode(mu, sigma, tau=0.83)
        assert calls > 0, "the registered family decode must invoke the override itself"
        calls_after_decode = calls
        reference = pb.reference_decode(mu, sigma, tau=0.83)
        assert calls > calls_after_decode, "the reference oracle must be a separate functional call"
        torch.testing.assert_close(logits, reference, atol=5e-6, rtol=2e-5)
    finally:
        register_functional("renyi", override=True)(builtin)


def test_mixed_public_dtypes_and_nonactive_precision_policies_keep_generic_oracle():
    _, pb = _bank()
    mu, sigma, _ = _spd_inputs()
    _assert_generic_oracle(pb, mu.double(), sigma)

    previous = set_full_cov_kl_precision("fp64")
    try:
        _assert_generic_oracle(pb, mu, sigma)
    finally:
        set_full_cov_kl_precision(previous)

    previous = prior_bank.set_decode_av_precision("fp64")
    try:
        _assert_generic_oracle(pb, mu, sigma)
    finally:
        prior_bank.set_decode_av_precision(previous)


def test_canonical_non_pd_query_is_uniform_and_excluded_like_ignore_index():
    """The analytic delegate must retain the public full-chunked invalid-SPD behavior."""
    _, pb = _bank(decode_unigram_prior=False)
    mu, sigma, targets = _spd_inputs()
    sigma = sigma.clone()
    sigma[0, 1] = -5.0 * torch.eye(4, device=DEVICE)

    logits = pb.decode(mu, sigma)
    assert torch.isfinite(logits).all()
    assert torch.equal(logits[0, 1], logits[0, 1, 0].expand_as(logits[0, 1]))
    automatic = pb.decode_ce_family_chunked(mu, sigma, targets)
    ignored = targets.clone()
    ignored[0, 1] = -100
    explicit = pb.decode_ce_family_chunked(mu, sigma, ignored)
    assert torch.equal(automatic, explicit)


def test_same_name_family_override_then_restore_falls_closed_to_generic_logits_and_ce():
    """A bank built under a diagonal same-name override falls closed to the generic logits/CE."""
    builtin = get_family("gaussian_full")

    class DiagonalClaimingFullGaussian(FullGaussian):
        cov_kind = "diagonal"

    register_family("gaussian_full", override=True)(DiagonalClaimingFullGaussian)
    try:
        _, pb = _bank(untie_decode_bank=False)
    finally:
        register_family("gaussian_full", override=True)(builtin)

    mu, sigma, targets = _spd_inputs()
    logits = pb.decode(mu, sigma)
    reference = pb.reference_decode(mu, sigma)
    torch.testing.assert_close(logits, reference, atol=5e-6, rtol=2e-5)
    fused = pb.decode_ce_family_chunked(mu, sigma, targets)
    valid = targets != -100
    local_targets = targets.clamp_min(0)
    expected_per_position = torch.logsumexp(reference, dim=-1) - reference.gather(
        -1, local_targets.unsqueeze(-1),
    ).squeeze(-1)
    expected = expected_per_position[valid].mean()
    torch.testing.assert_close(fused, expected, atol=5e-6, rtol=2e-5)


def test_legacy_custom_fused_decoder_keeps_signature_when_stats_are_requested():
    model, _ = _bank(decode_mode="full_chunked", decode_unigram_prior=False)
    original = prior_bank.get_decode_registration("full_chunked")
    calls = []

    def legacy_fused(pb, mu_q, sigma_q, targets):
        calls.append(True)
        return original.fused_ce(pb, mu_q, sigma_q, targets)

    prior_bank.register_decode(
        "full_chunked",
        supports_full=True,
        supports_chunked=True,
        fused_ce=legacy_fused,
        family_consistent=original.family_consistent,
        covariance_kinds=original.covariance_kinds,
        can_omit_base_mean=original.can_omit_base_mean,
        can_omit_base_variance=original.can_omit_base_variance,
        override=True,
    )(original.callable)
    try:
        tokens = torch.randint(0, model.cfg.vocab_size, (2, 3), device=DEVICE)
        targets = torch.randint(0, model.cfg.vocab_size, (2, 3), device=DEVICE)
        targets[0, 1] = -100
        _, _, stats = model(tokens, targets, return_decode_stats=True)
        assert isinstance(stats, prior_bank.DecodeCEResult)
        assert int(stats.scored_tokens) == 5
        assert calls == [True]
    finally:
        # Canonical projected construction pins the complete import-time registration identity,
        # not only equal metadata/callables. Restore that exact object so this compatibility probe
        # cannot poison later registry-identity tests in the same process.
        prior_bank._DECODERS["full_chunked"] = original
