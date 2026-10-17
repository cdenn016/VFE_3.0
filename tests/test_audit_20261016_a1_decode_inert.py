r"""Inert decode precision knobs under the expected-likelihood decode (audit 2026-10-16, F36).

``decode_av_precision`` (the precision of the :math:`a_v = \sum_k 1/\sigma_{v,k}` reduction) and
``decode_ranking_fp64_escalation`` (the ranking guard) are read only by the ``diagonal_chunked`` and ``full_chunked``
KL decoders. ``expected_likelihood_chunked`` scores
:math:`-\tfrac{1}{2\tau}\sum_k[(\mu_q-\mu_v)^2/(\sigma_q+\sigma_v) + \log(\sigma_q+\sigma_v)]` with neither, so
``inactive_fields`` must mark both there, as it already did for the linear decode. Tiny CPU model.
"""
from __future__ import annotations

import warnings

import pytest
import torch

from tests.profiles import tiny_config
from vfe3.model.model import VFEModel

_FIELDS = ("decode_av_precision", "decode_ranking_fp64_escalation")


@pytest.fixture(autouse=True)
def _quiet():
    torch.set_num_threads(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.mark.parametrize("decode_mode, inert", [
    ("linear", True), ("expected_likelihood_chunked", True), ("diagonal_chunked", False),
])
def test_the_precision_knobs_are_marked_inert_exactly_off_the_kl_decoders(decode_mode: str, inert: bool) -> None:
    inactive = tiny_config(decode_mode=decode_mode).inactive_fields()
    for name in _FIELDS:
        assert (name in inactive) is inert, name
    full = tiny_config(decode_mode="full_chunked", family="gaussian_full").inactive_fields()
    assert not set(_FIELDS) & set(full)


def _loss_and_grads(**overrides) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    torch.manual_seed(0)
    model = VFEModel(tiny_config(decode_mode="expected_likelihood_chunked", **overrides))
    tokens = torch.tensor([[0, 3, 5], [6, 1, 2]])
    _, loss, _ = model(tokens, tokens)
    loss.backward()
    grads = [p.grad.clone() for p in model.parameters() if p.grad is not None]
    model.eval()
    with torch.no_grad():
        logits = model(tokens)                                   # (B, N, V)
    return loss.detach(), grads, logits


@pytest.mark.parametrize("override", [dict(decode_av_precision="fp64"), dict(decode_ranking_fp64_escalation=True)])
def test_the_expected_likelihood_decode_reads_neither_knob(override: dict) -> None:
    loss, grads, logits = _loss_and_grads()
    loss_x, grads_x, logits_x = _loss_and_grads(**override)
    assert torch.equal(loss, loss_x)
    assert len(grads) == len(grads_x) and all(torch.equal(a, b) for a, b in zip(grads, grads_x))
    assert torch.equal(logits, logits_x)
