r"""The encoded covariance a decode caller passes under ``decode_covariance='encoded'`` (audit 2026-10-16 F64).

``forward_beliefs`` decodes with ``encoded_sigma`` set to the stack-input covariance. A caller that decodes a stored
belief again (the inference bank of ``vfe3.viz.extract``, the logit lens of ``vfe3.semantics.contextual``) has to
pass the same tensor, which a captured forward already carries as ``capture['prior'].sigma`` and as the diagnostic
capture's ``initial_belief.sigma``. Pinned here: both are the tensor the forward decoded with, so a re-decode that
passes either reproduces the forward logits exactly, and a re-decode that passes neither is refused with a message
naming the remedy. Tiny CPU models only (K = 4, two heads of 2, N = 3, vocab 7).
"""
from typing import Any

import pytest
import torch

from tests.profiles import tiny_config
from vfe3.model.model import VFEModel

_TOKENS = torch.tensor([[1, 4, 2], [5, 3, 6]])

_ROUTES = {
    "diagonal": dict(decode_mode="diagonal_chunked"),
    "diagonal_reference": dict(decode_mode="diagonal_chunked", decode_coordinates="reference"),
    "full": dict(decode_mode="full_chunked", family="gaussian_full"),
}


def _model(**overrides: Any) -> VFEModel:
    values = dict(n_layers=2, sigma_init=2.0, phi_scale=0.3, decode_covariance="encoded")
    values.update(overrides)
    torch.manual_seed(0)
    model = VFEModel(tiny_config(**values)).eval()
    with torch.no_grad():
        # Unequal decode variances, so the decoded covariance reaches the logits through the trace term.
        model.prior_bank.sigma_log_embed.uniform_(-0.5, 1.0, generator=torch.Generator().manual_seed(1))
    return model


@pytest.mark.parametrize("route", sorted(_ROUTES))
def test_captured_stack_input_reproduces_the_forward_decode(route: str) -> None:
    model = _model(**_ROUTES[route])
    capture: dict = {"diagnostic": {}}
    with torch.no_grad():
        belief, logits = model.forward_beliefs(_TOKENS, return_logits=True, capture=capture)
        frame = capture.get("canonical_frame")
        assert (frame is not None) == (model.cfg.decode_coordinates == "reference")
        for encoded_sigma in (capture["prior"].sigma, capture["diagnostic"]["initial_belief"].sigma):
            again = model._decode_belief_with_context(belief.mu, belief.sigma, canonical_frame=frame,
                                                      encoded_sigma=encoded_sigma)
            torch.testing.assert_close(again, logits, rtol=0.0, atol=0.0)
        # The belief's own covariance is a different reading: the stack moved it.
        belief_reading = model._decode_belief_with_context(belief.mu, belief.sigma, canonical_frame=frame,
                                                           encoded_sigma=belief.sigma)
        assert not torch.equal(belief_reading, logits)


def test_a_decode_without_the_stack_input_names_the_remedy() -> None:
    model = _model(**_ROUTES["diagonal"])
    with torch.no_grad():
        belief, _ = model.forward_beliefs(_TOKENS)
        with pytest.raises(ValueError, match=r"encoded covariance: pass encoded_sigma"):
            model._decode_belief_with_context(belief.mu, belief.sigma)
