r"""float64 escalation of the moment-matched covariance update (``predictive_covariance_fp64_escalation``, 2026-10-15).

diag(U C^att U^T) leaves the reference coordinates through U, so float32 rounding of C^att is amplified by about
kappa(U)^2. On the K=160 block_glk run a variance came out at -5164 (about 1e-3 of its absolute term sum), and the
next layer's diagonal comparison route took its square root. Under the toggle a float32 update with any nonpositive
(or non-finite) variance, or a non-positive-definite full covariance, is recomputed whole in float64, cast back and
counted in its own counter (covariance_escalation_elements; audit 2026-10-15); a float64 failure raises. Off, the update is unchanged. The failing float32
update is injected so the trigger is exercised exactly; tiny CPU shapes only.
"""
from typing import Callable

import pytest
import torch

from tests.test_blockwise_moment_matched_20261004 import _case
from vfe3.config import VFE3Config
from vfe3.model.relational_operators import covariance_escalation_elements, reset_covariance_escalation_elements
from vfe3.relational_scoring import reset_source_recovery_elements, source_recovery_elements


def _poisoned(ops, poison_fp64: bool = False) -> Callable:
    r"""Wrap the update so a float32 call (and optionally a float64 call) returns one nonpositive variance."""
    original = ops._predictive_covariance_update

    def update(mu, sigma, *args):
        out = original(mu, sigma, *args)
        if out.dtype == torch.float32 or poison_fp64:
            out = out.clone()
            out.view(-1)[3] = -abs(float(out.view(-1)[3])) - 1.0
        return out
    ops._predictive_covariance_update = update
    return original


def _call(ops, mu, sigma, beta, frame, receive):
    return ops.predictive_covariance_update(mu, sigma, beta, frame, receive, floor_anchor=torch.ones_like(sigma))


def test_off_leaves_the_float32_update_unchanged() -> None:
    ops, mu, sigma, beta, frame, receive = _case(2, 2, False, (2,), torch.float32, seed=1)
    assert ops.predictive_covariance_fp64_escalation is False
    _poisoned(ops)
    assert bool((_call(ops, mu, sigma, beta, frame, receive) <= 0).any())


def test_nonpositive_float32_variance_is_recomputed_in_float64() -> None:
    ops, mu, sigma, beta, frame, receive = _case(2, 2, False, (2,), torch.float32, seed=1)
    ops.predictive_covariance_fp64_escalation = True
    _poisoned(ops)
    reset_covariance_escalation_elements()
    reset_source_recovery_elements()
    out = _call(ops, mu, sigma, beta, frame, receive)
    assert out.dtype == torch.float32 and bool((out > 0).all())
    reference = ops.predictive_covariance_update(mu.double(), sigma.double(), beta.double(), frame,
                                                 receive.double(), floor_anchor=torch.ones_like(sigma).double())
    torch.testing.assert_close(out[0], reference[0].float())     # the poisoned sequence, recomputed in float64
    assert covariance_escalation_elements() == mu.shape[-2]         # one sequence of N tokens
    assert source_recovery_elements() == 0                          # no longer mixed into the scorer counter


def test_healthy_float32_update_is_not_escalated() -> None:
    ops, mu, sigma, beta, frame, receive = _case(2, 2, False, (2,), torch.float32, seed=1)
    off = _call(ops, mu, sigma, beta, frame, receive)
    ops.predictive_covariance_fp64_escalation = True
    reset_covariance_escalation_elements()
    reset_source_recovery_elements()
    assert torch.equal(_call(ops, mu, sigma, beta, frame, receive), off)
    assert covariance_escalation_elements() == 0


def test_float64_failure_raises() -> None:
    ops, mu, sigma, beta, frame, receive = _case(2, 2, False, (2,), torch.float32, seed=1)
    ops.predictive_covariance_fp64_escalation = True
    _poisoned(ops, poison_fp64=True)
    with pytest.raises(ValueError, match="float64"):
        _call(ops, mu, sigma, beta, frame, receive)


def test_escalated_update_carries_gradients_to_float32_leaves() -> None:
    ops, mu, sigma, beta, frame, receive = _case(2, 2, False, (2,), torch.float32, seed=1)
    ops.predictive_covariance_fp64_escalation = True
    _poisoned(ops)
    mu, sigma = mu.requires_grad_(True), sigma.requires_grad_(True)
    _call(ops, mu, sigma, beta, frame, receive).sum().backward()
    assert sigma.grad.dtype == torch.float32 and bool(torch.isfinite(sigma.grad).all())
    assert bool(torch.isfinite(mu.grad).all())


def test_config_field_is_a_bool() -> None:
    assert VFE3Config().predictive_covariance_fp64_escalation is False
    with pytest.raises(ValueError, match="predictive_covariance_fp64_escalation"):
        VFE3Config(predictive_covariance_fp64_escalation=1)


def test_escalation_accepts_per_run_receiving_maps() -> None:
    r"""Operators that pass O' as a per-run list (the K=160 run's relational_operators_extra) escalate too."""
    ops, mu, sigma, beta, frame, receive = _case(2, 2, False, (2,), torch.float32, seed=1)
    ops.predictive_covariance_fp64_escalation = True
    _poisoned(ops)
    out = _call(ops, mu, sigma, beta, frame, [receive])
    reference = ops.predictive_covariance_update(mu.double(), sigma.double(), beta.double(), frame,
                                                 receive.double(), floor_anchor=torch.ones_like(sigma).double())
    torch.testing.assert_close(out[0], reference[0].float())


def test_only_the_failing_sequences_are_recomputed() -> None:
    r"""Healthy sequences keep their float32 values bit for bit; the failing one takes the float64 values."""
    from vfe3.contracts import CanonicalFrameContext
    ops, mu, sigma, beta, frame, receive = _case(2, 2, False, (3,), torch.float32, seed=4)
    healthy = _call(ops, mu, sigma, beta, frame, receive)
    ops.predictive_covariance_fp64_escalation = True
    original = ops._predictive_covariance_update
    poisoned_batch = {"rows": None}

    def update(mu_, sigma_, *args):
        out = original(mu_, sigma_, *args)
        if out.dtype == torch.float32:
            poisoned_batch["rows"] = out.shape[0]
            out = out.clone()
            out[1, 2, 0] = -1.0                                  # sequence 1 fails
        return out
    ops._predictive_covariance_update = update
    reset_covariance_escalation_elements()
    reset_source_recovery_elements()
    for shared in (False, True):
        f = (CanonicalFrameContext(frame.forward[:1], frame.inverse[:1], block_diagonal=True) if shared
             else frame)
        out = _call(ops, mu, sigma, beta, f, receive)
        ops._predictive_covariance_update = original
        reference = _call(ops, mu.double(), sigma.double(), beta.double(), f, receive.double())
        expect = _call(ops, mu, sigma, beta, f, receive)
        ops._predictive_covariance_update = update
        assert torch.equal(out[0], expect[0]) and torch.equal(out[2], expect[2])
        torch.testing.assert_close(out[1], reference[1].float())
    assert covariance_escalation_elements() == 2 * mu.shape[-2]  # one sequence per call

