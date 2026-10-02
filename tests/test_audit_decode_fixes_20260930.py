"""Tiny CPU regressions for numerical and retained-work decoder repairs."""

import math
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

import vfe3.model.prior_bank as bank_module
from vfe3.model.prior_bank import PriorBank, _decode_av, _decode_av_lhs

MODES = ("diagonal_chunked", "full_chunked", "expected_likelihood_chunked", "family_chunked")


def _bank(mode: str, checkpoint: str = "never", precision: str = "fp32") -> PriorBank:
    full = mode == "full_chunked"
    return PriorBank(
        7, 2, 3, family="gaussian_full" if full else "gaussian_diagonal",
        diagonal_covariance=not full, decode_mode=mode, decode_chunk_size=4,
        decode_tau=1.0,
        decode_ce_checkpoint=checkpoint, decode_av_precision=precision,
        decode_ranking_fp64_escalation=False,
    )


def _query(mode: str, mean: float = 0.2) -> tuple[torch.Tensor, torch.Tensor]:
    mu_q = torch.full((1, 3, 2), mean, requires_grad=True)
    sigma_q = (torch.eye(2).expand(1, 3, 2, 2).clone() if mode == "full_chunked"
               else torch.ones_like(mu_q))
    return mu_q, sigma_q.requires_grad_(True)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("checkpoint", ["always", "never"])
def test_large_equal_scores_preserve_log_normalizer(mode: str, checkpoint: str) -> None:
    pb = _bank(mode, checkpoint)
    with torch.no_grad():
        pb.mu_embed.zero_()
        pb.sigma_log_embed.fill_(math.log(1e-4))
    mu_q, sigma_q = _query(mode, 3000.0)
    ce = getattr(pb, "decode_ce_" + mode)(mu_q, sigma_q, torch.tensor([[0, 4, 6]]))
    torch.testing.assert_close(ce, ce.new_tensor(math.log(7)), atol=3e-7, rtol=0)
    gradient = torch.autograd.grad(ce, mu_q)[0]
    # fp32 still rounds the cancelling softmax covectors; exclude the former O(1e8) error.
    assert gradient.abs().max() < 32.0


@pytest.mark.parametrize("mode", MODES)
def test_target_shape_cannot_broadcast(mode: str) -> None:
    pb = _bank(mode)
    mu_q, sigma_q = _query(mode)
    with pytest.raises(ValueError, match="shape"):
        getattr(pb, "decode_ce_" + mode)(mu_q.expand(2, -1, -1),
            sigma_q.expand(2, *sigma_q.shape[1:]), torch.tensor([[0, 1, 6]]))


def test_fp64_query_factor_is_built_after_promotion() -> None:
    sq = torch.tensor([[[0.1234567]]])
    mc_q = torch.tensor([[[10.000123]]])
    mc_v = torch.tensor([[0.0], [0.000017], [-0.000013]])
    inv_v = torch.tensor([[10000.0], [10000.1], [9999.9]])
    lsum = -inv_v.log().sum(-1)
    direct = _decode_av(sq, mc_q, mc_v, inv_v, lsum, precision="fp64")
    hoisted = _decode_av(sq, mc_q, mc_v, inv_v, lsum,
                        lhs=_decode_av_lhs(sq, mc_q), precision="fp64")
    assert torch.equal(direct, hoisted)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("checkpoint", ["always", "never"])
def test_packed_rows_match_dense_values_gradients_and_counts(mode: str, checkpoint: str) -> None:
    torch.manual_seed(93)
    pb = _bank(mode, checkpoint)
    mu_q, sigma_q = _query(mode)
    targets = torch.tensor([[0, -100, 6]])
    dense = F.cross_entropy(pb.decode(mu_q, sigma_q).reshape(-1, 7), targets.flatten())
    result = getattr(pb, "decode_ce_" + mode)(mu_q, sigma_q, targets, return_stats=True)
    torch.testing.assert_close(result.ce, dense, atol=3e-6, rtol=3e-6)
    assert int(result.scored_tokens) == 2
    assert int(result.excluded_tokens) == 1
    inputs = (mu_q, sigma_q, pb.mu_embed, pb.sigma_log_embed, pb.decode_log_scale)
    actual = torch.autograd.grad(result.ce, inputs)
    expected = torch.autograd.grad(dense, inputs)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, atol=3e-6, rtol=3e-5)
    assert torch.equal(actual[0][:, 1], torch.zeros_like(actual[0][:, 1]))


@pytest.mark.parametrize("mode", ["diagonal_chunked", "full_chunked"])
def test_ignored_rows_are_removed_before_scoring(mode: str) -> None:
    pb = _bank(mode)
    mu_q, sigma_q = _query(mode)
    with patch.object(bank_module, "_decode_av", wraps=bank_module._decode_av) as score:
        getattr(pb, "decode_ce_" + mode)(mu_q, sigma_q, torch.tensor([[0, -100, 6]]))
    assert {call.args[0].shape[:2] for call in score.call_args_list} == {(1, 2)}


@pytest.mark.parametrize("mode", MODES)
def test_all_ignored_remains_a_connected_zero(mode: str) -> None:
    pb = _bank(mode, "always")
    mu_q, sigma_q = _query(mode)
    result = getattr(pb, "decode_ce_" + mode)(mu_q, sigma_q, torch.full((1, 3), -100), return_stats=True)
    assert result.ce.item() == 0.0
    result.ce.backward()
    assert torch.equal(mu_q.grad, torch.zeros_like(mu_q))
    assert torch.equal(sigma_q.grad, torch.zeros_like(sigma_q))
    assert int(result.excluded_tokens) == 3


@pytest.mark.parametrize("mode", ["diagonal_chunked", "full_chunked"])
def test_inference_without_escalation_does_not_build_bounds(mode: str) -> None:
    pb = _bank(mode)
    mu_q, sigma_q = _query(mode)
    with patch.object(bank_module, "_decode_av", wraps=bank_module._decode_av) as score:
        logits = pb.decode(mu_q, sigma_q)
    assert torch.isfinite(logits).all()
    assert all(not call.kwargs.get("return_error_bound", False) for call in score.call_args_list)


@pytest.mark.parametrize("mode", ["diagonal_chunked", "full_chunked", "expected_likelihood_chunked"])
def test_auto_checkpoint_considers_whole_vocabulary(mode: str) -> None:
    pb = _bank(mode, "auto_aggregate")
    pb.decode_chunk_size = 2
    mu_q, sigma_q = _query(mode)
    # Each two-class workspace is 48 bytes, but the retained vocabulary is 168 bytes.
    with patch.object(bank_module, "DECODE_CE_CHECKPOINT_AUTO_BYTES", 100), \
         patch.object(bank_module._checkpoint, "checkpoint", wraps=bank_module._checkpoint.checkpoint) as checkpoint:
        getattr(pb, "decode_ce_" + mode)(mu_q, sigma_q, torch.tensor([[0, 4, 6]]))
    assert checkpoint.call_count == 4


@pytest.mark.parametrize("mode", ["diagonal_chunked", "full_chunked"])
def test_fixed_decode_scope_reuses_factors_and_releases_on_exit(mode: str) -> None:
    pb = _bank(mode)
    mu_q, sigma_q = _query(mode)
    with torch.no_grad(), patch.object(pb, "_prepare_decode_bank_factors", wraps=pb._prepare_decode_bank_factors) as prepare:
        expected = pb.decode(mu_q, sigma_q)
        assert prepare.call_count == 1
        with pb.fixed_decode_bank():
            for _ in range(3):
                assert torch.equal(pb.decode(mu_q, sigma_q), expected)
            assert prepare.call_count == 2
        assert not hasattr(pb, "_fixed_decode_bank_cache")
        pb.mu_embed.add_(0.1)
        changed = pb.decode(mu_q, sigma_q)
        assert prepare.call_count == 3
        assert not torch.equal(changed, expected)
    # The transient cache is never used by training or retained as checkpoint state.
    with pb.fixed_decode_bank():
        ce = getattr(pb, "decode_ce_" + mode)(mu_q, sigma_q, torch.tensor([[0, 4, 6]]))
    ce.backward()
    assert pb.mu_embed.grad is not None


def test_fixed_decode_scope_releases_on_exception() -> None:
    pb = _bank("diagonal_chunked")
    mu_q, sigma_q = _query("diagonal_chunked")
    with torch.no_grad(), pytest.raises(RuntimeError, match="scope"):
        with pb.fixed_decode_bank():
            pb.decode(mu_q, sigma_q)
            raise RuntimeError("scope")
    assert not hasattr(pb, "_fixed_decode_bank_cache")

@pytest.mark.parametrize("mode", ["full_chunked", "expected_likelihood_chunked", "family_chunked"])
@pytest.mark.parametrize("channel", ["mu", "sigma"])
def test_nonlinear_gradient_matches_directional_finite_difference(mode: str, channel: str) -> None:
    pb = _bank(mode, "never", "fp64").double()
    mu_q, sigma_q = _query(mode)
    mu_q = mu_q.detach().double().requires_grad_(True)
    sigma_q = sigma_q.detach().double().requires_grad_(True)
    targets = torch.tensor([[0, -100, 6]])
    query = mu_q if channel == "mu" else sigma_q
    fn = getattr(pb, "decode_ce_" + mode)
    loss = fn(mu_q, sigma_q, targets)
    direction = torch.zeros_like(query)
    direction.flatten()[0] = 1.0
    analytic = (torch.autograd.grad(loss, query)[0] * direction).sum()
    h = 1e-5
    with torch.no_grad():
        query.add_(h * direction)
        plus = fn(mu_q, sigma_q, targets)
        query.sub_(2 * h * direction)
        minus = fn(mu_q, sigma_q, targets)
        query.add_(h * direction)
    torch.testing.assert_close(analytic, (plus - minus) / (2 * h), atol=1e-8, rtol=1e-5)


@pytest.mark.parametrize("generic_family", [False, True])
def test_full_covariance_invalid_and_ignored_rows_keep_original_statistics(generic_family: bool) -> None:
    pb = _bank("full_chunked")
    if generic_family:
        pb.decode_mode = "family_chunked"
        pb.renyi_order = 0.5  # exercises the functional route, without the analytic KL delegate
    mu_q, sigma_q = _query("full_chunked")
    with torch.no_grad():
        sigma_q[:, 0] = -5 * torch.eye(2)
    targets = torch.tensor([[0, -100, 6]])
    fn = pb.decode_ce_family_chunked if generic_family else pb.decode_ce_full_chunked
    result = fn(mu_q, sigma_q, targets, return_stats=True)
    assert int(result.scored_tokens) == 1
    assert int(result.excluded_tokens) == 2
    result.ce.backward()
    assert torch.isfinite(mu_q.grad).all()
    assert torch.isfinite(sigma_q.grad).all()


@pytest.mark.parametrize("mode", ["diagonal_chunked", "full_chunked"])
def test_auto_retention_counts_fp64_scoring_and_decoder_only_gradients(mode: str) -> None:
    pb = _bank(mode, "auto_aggregate", "fp64")
    mu_q, sigma_q = _query(mode)
    mu_q = mu_q.detach()
    sigma_q = sigma_q.detach()
    # fp32 query storage is 168 bytes in total; fp64 score storage is 336 bytes.
    with patch.object(bank_module, "DECODE_CE_CHECKPOINT_AUTO_BYTES", 200), \
         patch.object(bank_module._checkpoint, "checkpoint", wraps=bank_module._checkpoint.checkpoint) as checkpoint:
        ce = getattr(pb, "decode_ce_" + mode)(mu_q, sigma_q, torch.tensor([[0, 4, 6]]))
    assert checkpoint.call_count == 2
    ce.backward()
    assert torch.isfinite(pb.mu_embed.grad).all()


@pytest.mark.parametrize("mode", ["diagonal_chunked", "full_chunked"])
def test_reference_frame_pullback_precedes_row_packing(mode: str) -> None:
    from vfe3.contracts import CanonicalFrameContext
    pb = _bank(mode)
    pb.decode_coordinates = "reference"
    mu_q, sigma_q = _query(mode)
    U = torch.diag(torch.tensor([2.0, 0.5])).expand(1, 3, 2, 2).clone()
    frame = CanonicalFrameContext(U, torch.linalg.inv(U))
    targets = torch.tensor([[0, -100, 6]])
    dense = F.cross_entropy(pb.decode(mu_q, sigma_q, canonical_frame=frame).reshape(-1, 7), targets.flatten())
    fused = getattr(pb, "decode_ce_" + mode)(mu_q, sigma_q, targets, canonical_frame=frame)
    torch.testing.assert_close(fused, dense, rtol=3e-6, atol=3e-6)
    for actual, expected in zip(torch.autograd.grad(fused, (mu_q, sigma_q)), torch.autograd.grad(dense, (mu_q, sigma_q))):
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
