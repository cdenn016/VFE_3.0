r"""``decode_covariance``: the covariance the prior-bank decoder reads (2026-10-11).

``"belief"`` decodes the stack's final :math:`\Sigma_q`, ``"encoded"`` the stack's input covariance and
``"constant"`` :math:`\sigma_{\mathrm{init}}I`. Tiny CPU models only. Pins that the training and inference seams
read the same selected covariance, that ``"encoded"`` equals ``"belief"`` when no seam changes the covariance,
that the linear readout marks the field inert, and that an unknown value is refused.
"""

import pytest
import torch
import torch.nn.functional as F

from tests.profiles import tiny_config
from vfe3.model.model import VFEModel

_TOKENS = torch.tensor([[1, 4, 2], [5, 3, 6]])
_TARGETS = torch.tensor([[4, 2, 5], [3, 6, 1]])


def _model(**overrides: object) -> VFEModel:
    values = dict(decode_mode="diagonal_chunked", n_layers=2, sigma_init=2.0)
    values.update(overrides)
    torch.manual_seed(0)
    model = VFEModel(tiny_config(**values))
    with torch.no_grad():
        # Unequal decode variances: with equal sigma_v the trace term tr(Sigma_v^-1 Sigma_q) is common to every v
        # and cancels in the softmax, so the decoded covariance would not reach the logits at all.
        model.prior_bank.sigma_log_embed.uniform_(-0.5, 1.0, generator=torch.Generator().manual_seed(1))
    return model


def _loss_and_grads(model: VFEModel) -> tuple:
    _, loss, _ = model(_TOKENS, _TARGETS)
    loss.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    return loss.detach(), grads


@pytest.mark.parametrize("mode", ["belief", "encoded", "constant"])
def test_training_and_inference_seams_read_the_selected_covariance(mode: str) -> None:
    model = _model(decode_covariance=mode)
    with torch.no_grad():
        belief, logits = model.forward_beliefs(_TOKENS, return_logits=True)
        encoded = model.prior_bank.encode(_TOKENS).sigma
        expected_sigma = {"belief": belief.sigma, "encoded": encoded,
                          "constant": torch.full_like(belief.sigma, 2.0)}[mode]
        expected = model.prior_bank.decode(belief.mu, expected_sigma)
        _, _, ce = model(_TOKENS, _TARGETS)
    torch.testing.assert_close(logits, expected, rtol=0.0, atol=0.0)
    dense = F.cross_entropy(expected.reshape(-1, expected.shape[-1]), _TARGETS.reshape(-1))
    torch.testing.assert_close(ce, dense, rtol=1e-5, atol=1e-5)


def test_the_three_readings_differ_under_a_moment_matched_covariance() -> None:
    losses = {mode: _loss_and_grads(_model(decode_covariance=mode))[0] for mode in ("belief", "encoded", "constant")}
    assert len({float(v) for v in losses.values()}) == 3


def test_encoded_equals_belief_when_no_seam_changes_the_covariance() -> None:
    loss_b, grads_b = _loss_and_grads(_model(decode_covariance="belief", predictive_covariance="passthrough"))
    loss_e, grads_e = _loss_and_grads(_model(decode_covariance="encoded", predictive_covariance="passthrough"))
    assert torch.equal(loss_b, loss_e)
    assert grads_b.keys() == grads_e.keys()
    for name in grads_b:
        assert torch.equal(grads_b[name], grads_e[name]), name


def test_constant_reads_sigma_init_times_identity_on_the_full_family() -> None:
    model = _model(decode_covariance="constant", family="gaussian_full", decode_mode="full_chunked")
    with torch.no_grad():
        belief, logits = model.forward_beliefs(_TOKENS, return_logits=True)
        eye = torch.eye(belief.mu.shape[-1]).expand(*belief.mu.shape[:-1], -1, -1)
        expected = model.prior_bank.decode(belief.mu, 2.0 * eye)
    torch.testing.assert_close(logits, expected, rtol=0.0, atol=0.0)


def test_encoded_without_the_stack_input_is_refused() -> None:
    model = _model(decode_covariance="encoded")
    belief, _ = model.forward_beliefs(_TOKENS)
    with pytest.raises(ValueError, match="encoded covariance"):
        model._decode_belief_with_context(belief.mu, belief.sigma)


def test_linear_readout_marks_the_field_inert_and_unknown_values_are_refused() -> None:
    assert "decode_covariance" in tiny_config(decode_mode="linear").inactive_fields()
    assert "decode_covariance" not in tiny_config(decode_mode="diagonal_chunked").inactive_fields()
    with pytest.raises(ValueError, match="decode_covariance"):
        tiny_config(decode_mode="diagonal_chunked", decode_covariance="prior")
