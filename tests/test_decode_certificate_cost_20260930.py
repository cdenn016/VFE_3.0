"""Unused decode certificates must not allocate bounds or run extra GEMMs."""

from collections import Counter
from typing import Any, Optional

import pytest
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from vfe3.model.prior_bank import PriorBank, _decode_av


class _OperationCounts(TorchDispatchMode):
    """Count actual CPU operations without replacing any decoder arithmetic."""

    def __init__(self) -> None:
        super().__init__()
        self.counts: Counter[str] = Counter()

    def __torch_dispatch__(
        self,
        func:   Any,
        types:  tuple[type, ...],
        args:   tuple[Any, ...]          = (),
        kwargs: Optional[dict[str, Any]] = None,
    ) -> Any:
        self.counts[str(func)] += 1
        return func(*args, **(kwargs or {}))

    @property
    def gemms(self) -> int:
        return self.counts["aten.mm.default"] + self.counts["aten.bmm.default"]


def _bank(
    checkpoint: str,
    precision:  str  = "fp32",
    evidence:   bool = False,
) -> PriorBank:
    torch.manual_seed(30)
    pb = PriorBank(
        7, 4, 4, decode_mode="diagonal_chunked", decode_chunk_size=3,
        decode_av_precision=precision, decode_ce_checkpoint=checkpoint,
        decode_ranking_fp64_escalation=False, irrep_dims=[2, 2],
        use_priorbank_head_evidence_mixer=evidence,
        decode_unigram_prior=evidence,
    )
    with torch.no_grad():
        pb.mu_embed.copy_(torch.randn_like(pb.mu_embed) * 0.3)
        pb.sigma_log_embed.copy_(torch.randn_like(pb.sigma_log_embed) * 0.2)
        if evidence:
            pb.head_evidence_logits.copy_(torch.tensor([0.4, -0.3]))
            pb.set_unigram_log_prior(torch.arange(7, dtype=torch.float32) * 3.0)
    return pb


@pytest.mark.parametrize("precision", ["fp32", "fp64"])
@pytest.mark.parametrize("return_error_bound", [False, True])
def test_decode_av_only_computes_a_bound_when_requested(
    precision:         str,
    return_error_bound: bool,
) -> None:
    generator = torch.Generator().manual_seed(31)
    sq = torch.rand(1, 2, 4, generator=generator) + 0.5
    mc_q = torch.randn(1, 2, 4, generator=generator)
    mc_v = torch.randn(7, 4, generator=generator)
    inv_v = torch.rand(7, 4, generator=generator) + 0.5
    lsum = -inv_v.log().sum(-1)
    with _OperationCounts() as operations:
        result = _decode_av(
            sq, mc_q, mc_v, inv_v, lsum,
            return_error_bound=return_error_bound, precision=precision,
        )
    assert operations.gemms == (2 if return_error_bound and precision == "fp32" else 1)
    assert operations.counts["aten.zeros_like.default"] == int(
        return_error_bound and precision == "fp64")
    value = result[0] if return_error_bound else result
    dtype = torch.float64 if precision == "fp64" else torch.float32
    expected = (
        (sq.to(dtype).unsqueeze(-2)
         + (mc_q.to(dtype).unsqueeze(-2) - mc_v.to(dtype)).square())
        * inv_v.to(dtype)
    ).sum(-1) + lsum.to(dtype)
    torch.testing.assert_close(value, expected, rtol=2e-6, atol=2e-6)
    if return_error_bound:
        assert torch.isfinite(result[1]).all()


@pytest.mark.parametrize("checkpoint", ["auto", "always", "never"])
@pytest.mark.parametrize("escalation", [False, True])
def test_diagonal_ce_skips_unused_ranking_work(
    checkpoint: str,
    escalation: bool,
) -> None:
    pb = _bank(checkpoint)
    pb.decode_ranking_fp64_escalation = escalation
    mu_q = torch.full((1, 2, 4), 0.1, requires_grad=True)
    sigma_q = torch.ones_like(mu_q, requires_grad=True)
    with _OperationCounts() as operations:
        ce = pb.decode_ce_diagonal_chunked(mu_q, sigma_q, torch.tensor([[0, 6]]))
    # Three vocabulary chunks need three score GEMMs and no ranking certificate.
    assert operations.gemms == 3
    assert operations.counts["aten.nextafter.default"] == 0
    assert operations.counts["aten.argmax.default"] == 0
    assert torch.isfinite(ce)
    ce.backward()
    assert torch.isfinite(mu_q.grad).all()
    assert torch.isfinite(sigma_q.grad).all()


@pytest.mark.parametrize("checkpoint", ["always", "never"])
@pytest.mark.parametrize("precision", ["fp32", "fp64"])
@pytest.mark.parametrize("evidence", [False, True])
def test_diagonal_ce_value_and_gradients_match_dense_decode(
    checkpoint: str,
    precision:  str,
    evidence:   bool,
) -> None:
    pb = _bank(checkpoint, precision, evidence)
    mu_q = torch.full((1, 3, 4), 0.1, requires_grad=True)
    sigma_q = torch.full_like(mu_q, 0.8, requires_grad=True)
    targets = torch.tensor([[0, -100, 6]])
    logits = pb.decode(mu_q, sigma_q)
    dense_ce = F.cross_entropy(logits.reshape(-1, 7), targets.reshape(-1))
    fused_ce = pb.decode_ce_diagonal_chunked(mu_q, sigma_q, targets)
    inputs = (mu_q, sigma_q, pb.mu_embed, pb.sigma_log_embed, pb.decode_log_scale)
    if evidence:
        inputs += (pb.head_evidence_logits,)
    dense_gradients = torch.autograd.grad(dense_ce, inputs)
    fused_gradients = torch.autograd.grad(fused_ce, inputs)
    torch.testing.assert_close(fused_ce, dense_ce, rtol=2e-6, atol=2e-6)
    for fused, dense in zip(fused_gradients, dense_gradients):
        torch.testing.assert_close(fused, dense, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("query_channel", ["mu", "sigma"])
def test_diagonal_ce_query_gradient_matches_finite_difference(query_channel: str) -> None:
    pb = _bank("never", "fp64").double()
    mu_q = torch.full((1, 2, 4), 0.1, dtype=torch.float64, requires_grad=True)
    sigma_q = torch.full_like(mu_q, 0.8, requires_grad=True)
    targets = torch.tensor([[0, 6]])
    query = mu_q if query_channel == "mu" else sigma_q
    ce = pb.decode_ce_diagonal_chunked(mu_q, sigma_q, targets)
    gradient = torch.autograd.grad(ce, query)[0][0, 0, 0]
    with torch.no_grad():
        query[0, 0, 0] += 1e-5
        plus = pb.decode_ce_diagonal_chunked(mu_q, sigma_q, targets)
        query[0, 0, 0] -= 2e-5
        minus = pb.decode_ce_diagonal_chunked(mu_q, sigma_q, targets)
    torch.testing.assert_close(gradient, (plus - minus) / 2e-5, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("checkpoint", ["always", "never"])
def test_near_ties_keep_the_streamed_float32_ce(checkpoint: str) -> None:
    # The candidate-refined guard certifies only the logits ranking; the training CE never reads it.
    pb = _bank(checkpoint)
    pb.decode_ranking_fp64_escalation = True
    with torch.no_grad():
        pb.mu_embed.zero_()
        pb.sigma_log_embed.zero_()
    mu_q = torch.full((1, 2, 4), 0.1, requires_grad=True)
    sigma_q = torch.full_like(mu_q, 0.8, requires_grad=True)
    targets = torch.tensor([[0, 6]])
    with _OperationCounts() as operations:
        ce = pb.decode_ce_diagonal_chunked(mu_q, sigma_q, targets)
    assert operations.counts["aten.argmax.default"] == 0
    assert operations.counts["aten.nextafter.default"] == 0
    assert ce.dtype == torch.float32
    torch.testing.assert_close(ce, torch.tensor(7.0).log())


@pytest.mark.parametrize("checkpoint", ["always", "never"])
def test_diagonal_fast_ce_all_ignored_is_a_grad_connected_zero(checkpoint: str) -> None:
    pb = _bank(checkpoint, evidence=True)
    mu_q = torch.full((1, 2, 4), 0.1, requires_grad=True)
    sigma_q = torch.full_like(mu_q, 0.8, requires_grad=True)
    ce = pb.decode_ce_diagonal_chunked(mu_q, sigma_q, torch.full((1, 2), -100))
    assert ce.item() == 0.0
    ce.backward()
    assert torch.equal(mu_q.grad, torch.zeros_like(mu_q))
    assert torch.equal(sigma_q.grad, torch.zeros_like(sigma_q))


def test_full_ce_also_skips_unrequested_bound_gemms() -> None:
    pb = PriorBank(
        7, 4, 4, diagonal_covariance=False, family="gaussian_full",
        decode_mode="full_chunked", decode_chunk_size=3, decode_av_precision="fp32",
        decode_ce_checkpoint="never", decode_ranking_fp64_escalation=False,
    )
    mu_q = torch.full((1, 2, 4), 0.1, requires_grad=True)
    sigma_q = torch.eye(4).expand(1, 2, 4, 4).clone().requires_grad_(True)
    with _OperationCounts() as operations:
        ce = pb.decode_ce_full_chunked(mu_q, sigma_q, torch.tensor([[0, 6]]))
    assert operations.gemms == 3
    assert operations.counts["aten.nextafter.default"] == 0
    assert torch.isfinite(ce)
    ce.backward()
    assert torch.isfinite(mu_q.grad).all()
    assert torch.isfinite(sigma_q.grad).all()
