r"""Decode behavior on a non-PD Sigma_q (audit 2026-08-06 F31).

Two separate things live here, and the distinction is the point.

**Fixed.** The chunked reducers returned ``gathered * in_chunk_f`` to zero out positions whose
target is not in this chunk. With an -inf logit that is ``-inf * 0.0`` = NaN, so a degenerate
position poisoned ``target_logit`` through chunks it has no business contributing to. A mask must
not manufacture NaN under any contract; it is now a select, byte-identical for finite logits.

**Also fixed, as a decision.** ``logdet_q = -inf`` on a total Cholesky failure used to be a PINNED
cross-path parity contract (``test_full_cov_chunked_matches_dense_on_non_pd``, audit 2026-07-01 F6):
the dense and chunked paths had to agree that a degenerate position scores -inf. That contract was
itself the NaN generator -- an all--inf logit row is NaN under ``log_softmax`` on BOTH paths, and the
chunked ``logsumexp_v - target_logit`` is ``-inf - (-inf)``.

The resolution is EXCLUSION rather than a different sentinel: a position whose ``Sigma_q`` is not
positive definite has no valid likelihood, so it contributes nothing and leaves the denominator, and
the loss of tokens is visible through ``decode_logdet_fallback_elements``. Any finite sentinel would
have worked numerically -- ``per_pos`` is v-independent and cancels exactly -- but a sentinel
fabricates a score, and ``log V`` in particular reads as a plausible loss. Both paths moved
together: the fused kernels fold the mask into ``valid``, and the dense branch asks
``decode_degenerate_positions`` and marks the position ``ignore_index``. The F6 pin is restated in
terms of the exclusion, not deleted.

Pins: (a) the mask no longer manufactures NaN; (b) the degenerate row is finite, uniform, and
NaN-free under log_softmax; (c) the fused CE excludes the position from BOTH numerator and
denominator; (d) the event is counted; (e) the healthy path is untouched.
"""

import pytest
import torch

import vfe3.model.prior_bank as prior_bank
from vfe3.model.model import VFEModel
from vfe3.numerics import (
    decode_logdet_fallback_elements,
    reset_decode_logdet_fallback_elements,
)

from tests.test_amp import _tiny_model


@pytest.fixture(autouse=True)
def _reset():
    reset_decode_logdet_fallback_elements()
    yield
    reset_decode_logdet_fallback_elements()


def _bank() -> tuple[VFEModel, prior_bank.PriorBank]:
    """Use a local decoder policy and exactly representable, nonuniform prior means.

    Excluding rows packs the decoder batch. Random fp32 dot products and loss sums can change by
    one ULP when that shape changes; dyadic inputs and fp64 reduction keep the bitwise exclusion
    oracle independent of the process RNG and earlier tests' arithmetic policies.
    """
    model = _tiny_model(vocab_size=7, gauge_group="block_glk", n_heads=2,
                        family="gaussian_full", decode_mode="full_chunked",
                        decode_av_precision="fp64",
                        decode_ranking_fp64_escalation=False, decode_ce_checkpoint="auto",
                        fp32_matmul_precision="highest", full_cov_kl_precision="fp32_escalate")
    bank = model.prior_bank
    with torch.no_grad():
        row = torch.arange(bank.vocab_size) - bank.vocab_size // 2
        coord = torch.arange(bank.K) + 1
        bank._decode_mu_table().copy_(row[:, None] * coord[None, :] / 16)  # (V, K)
        bank._decode_sigma_log_table().zero_()                            # Sigma_v = I
    return model, bank


def _inputs(
    model: VFEModel,
    non_pd_at: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, N, K = 2, 5, model.cfg.embed_dim
    g = torch.Generator().manual_seed(0)
    mu = torch.randn(B, N, K, generator=g).mul(4).round().div(4)
    sigma = torch.eye(K).expand(B, N, K, K).clone()
    if non_pd_at is not None:
        sigma[non_pd_at] = -5.0 * torch.eye(K)      # fails every jitter round
    return mu, sigma, torch.randint(0, model.cfg.vocab_size, (B, N), generator=g)


# -- (a) the mask no longer manufactures NaN ---------------------------------------------------


def test_chunk_mask_selects_instead_of_multiplying():
    r"""-inf * 0.0 is NaN; the select is 0.0. Checked on the operation itself so the pin survives
    any refactor of the surrounding reducer."""
    gathered = torch.tensor([float("-inf"), 1.5])
    in_chunk_f = torch.tensor([0.0, 1.0])
    assert torch.isnan(gathered * in_chunk_f)[0], "the old form is supposed to produce NaN here"
    selected = torch.where(in_chunk_f > 0, gathered, torch.zeros_like(gathered))
    assert torch.equal(selected, torch.tensor([0.0, 1.5]))


def test_finite_logits_are_byte_identical_under_the_select():
    g = torch.Generator().manual_seed(1)
    gathered = torch.randn(64, generator=g)
    in_chunk_f = (torch.rand(64, generator=g) > 0.5).float()
    assert torch.equal(
        gathered * in_chunk_f,
        torch.where(in_chunk_f > 0, gathered, torch.zeros_like(gathered)))


# -- (b) the degenerate row is finite and informationless, on both paths -----------------------


def test_degenerate_row_is_uniform_and_finite_on_both_paths():
    r"""The restated F6 parity contract. If this ever changes, it must change on BOTH paths."""
    model, bank = _bank()
    mu, sigma, _ = _inputs(model, non_pd_at=(0, 2))
    tau = bank._tau_eff(None)
    dense = prior_bank._decode_full(bank, mu, sigma, tau)
    chunked = prior_bank._decode_full_chunked(bank, mu, sigma, tau)
    for name, out in (("dense", dense), ("chunked", chunked)):
        assert torch.isfinite(out[0, 2]).all(), name
        assert torch.equal(out[0, 2], torch.zeros_like(out[0, 2])), name
    assert torch.equal(dense[0, 2], chunked[0, 2])


def test_the_old_neg_inf_row_no_longer_nans_under_log_softmax():
    r"""The exact failure the -inf sentinel produced, now closed on the path that used to show it."""
    model, bank = _bank()
    mu, sigma, _ = _inputs(model, non_pd_at=(0, 2))
    dense = prior_bank._decode_full(bank, mu, sigma, bank._tau_eff(None))
    assert torch.isfinite(torch.log_softmax(dense[0, 2], dim=-1)).all()


# -- (c) the fused CE excludes the position from numerator AND denominator ---------------------


def test_fused_ce_excludes_the_degenerate_position():
    r"""Exclusion, not scoring: the CE must equal the CE of the same batch with that position
    already marked ignore_index, and must not be NaN."""
    model, bank = _bank()
    mu, sigma, targets = _inputs(model, non_pd_at=(0, 2))
    got = bank.decode_ce_full_chunked(mu, sigma, targets)
    masked = targets.clone()
    masked[0, 2] = -100
    reference = bank.decode_ce_full_chunked(mu, sigma, masked)
    assert torch.isfinite(got)
    assert torch.equal(got, reference)


def test_degenerate_position_leaves_the_denominator():
    r"""The token count really drops -- otherwise 'excluded' would just mean 'scored zero'."""
    model, bank = _bank()
    mu, sigma, targets = _inputs(model, non_pd_at=(0, 2))
    degenerate = bank.decode_degenerate_positions(sigma)
    assert degenerate is not None and int(degenerate.sum()) == 1 and bool(degenerate[0, 2])
    assert bank.decode_degenerate_positions(torch.rand(2, 5, model.cfg.embed_dim)) is None


# -- (d)/(e) the event is counted, and the healthy path is untouched ---------------------------


def test_fallback_is_counted():
    model, bank = _bank()
    mu, sigma, targets = _inputs(model, non_pd_at=(0, 2))
    bank.decode_ce_full_chunked(mu, sigma, targets)
    assert decode_logdet_fallback_elements() == 1


def test_healthy_path_counts_zero_and_stays_finite():
    model, bank = _bank()
    mu, sigma, targets = _inputs(model)
    assert torch.isfinite(bank.decode_ce_full_chunked(mu, sigma, targets))
    assert decode_logdet_fallback_elements() == 0


def test_model_forward_is_finite_and_counts_zero():
    model, _ = _bank()
    loss = model(torch.randint(0, model.cfg.vocab_size, (2, 5)),
                 torch.randint(0, model.cfg.vocab_size, (2, 5)))[1]
    assert torch.isfinite(loss)
    assert decode_logdet_fallback_elements() == 0


def test_nonfinite_rows_are_sanitized_excluded_counted_and_have_zero_gradient():
    r"""NaN and both infinities must leave the graph before any full-covariance arithmetic."""
    model, bank = _bank()
    mu, sigma, targets = _inputs(model)
    invalid_rows = ((0, 1), (0, 3), (1, 2), (1, 3))
    sigma[invalid_rows[0]] = float("nan")
    sigma[invalid_rows[1]] = float("inf")
    sigma[invalid_rows[2]] = float("-inf")
    sigma[invalid_rows[3]] = -5.0 * torch.eye(model.cfg.embed_dim)
    targets[1, 4] = -100
    mu.requires_grad_(True)
    sigma.requires_grad_(True)

    result = bank.decode_ce_full_chunked(mu, sigma, targets, return_stats=True)
    assert isinstance(result, prior_bank.DecodeCEResult)
    assert result.scored_tokens.dtype is torch.int64
    assert result.scored_tokens.device == sigma.device
    assert int(result.scored_tokens) == targets.numel() - len(invalid_rows) - 1
    assert torch.isfinite(result.ce)
    result.ce.backward()

    assert decode_logdet_fallback_elements() == len(invalid_rows)
    assert torch.isfinite(mu.grad).all()
    assert torch.isfinite(sigma.grad).all()
    assert mu.grad[1, 0].abs().sum() > 0  # A healthy scored row retains a likelihood gradient.
    for row in invalid_rows:
        assert torch.equal(mu.grad[row], torch.zeros_like(mu.grad[row]))
        assert torch.equal(sigma.grad[row], torch.zeros_like(sigma.grad[row]))

    safe_sigma = sigma.detach().clone()
    masked_targets = targets.clone()
    eye = torch.eye(model.cfg.embed_dim)
    for row in invalid_rows:
        safe_sigma[row] = eye
        masked_targets[row] = -100
    reset_decode_logdet_fallback_elements()
    reference = bank.decode_ce_full_chunked(mu.detach(), safe_sigma, masked_targets)
    assert torch.equal(result.ce.detach(), reference)
    assert decode_logdet_fallback_elements() == 0


def test_decoder_and_model_stats_are_opt_in_without_changing_default_returns():
    model, bank = _bank()
    mu, sigma, targets = _inputs(model)
    targets[0, 0] = -100

    scalar = bank.decode_ce_full_chunked(mu, sigma, targets)
    stats = bank.decode_ce_full_chunked(mu, sigma, targets, return_stats=True)
    assert isinstance(scalar, torch.Tensor)
    assert isinstance(stats, prior_bank.DecodeCEResult)
    assert torch.equal(stats.ce, scalar)
    assert int(stats.scored_tokens) == targets.numel() - 1

    tokens = torch.randint(0, model.cfg.vocab_size, targets.shape)
    default_result = model(tokens, targets)
    stats_result = model(tokens, targets, return_decode_stats=True)
    assert len(default_result) == len(stats_result) == 3
    assert isinstance(default_result[2], torch.Tensor)
    assert isinstance(stats_result[2], prior_bank.DecodeCEResult)
    assert torch.equal(stats_result[2].ce, default_result[2])
    assert int(stats_result[2].scored_tokens) == targets.numel() - 1
