r"""The ``research.json`` finite-difference decode-gradient probe runs in float64 (audit 2026-10-20 item 60).

In float32 at a step of 1e-3 its rounding floor is of order 0.3 relative for a correct gradient; on a float64 CPU copy
it resolves a correct adjoint to well below 1e-5 and still flags a wrong one. Tiny CPU models only.
"""
import math

import pytest
import torch

from tests.profiles import tiny_config
from vfe3.model.model import VFEModel
from vfe3.run_artifacts import _fd_gradient_check


def _model_and_batch():
    cfg = tiny_config()
    torch.manual_seed(0)
    model = VFEModel(cfg)                                                       # float32, as a trained run
    g = torch.Generator().manual_seed(7)
    batch = (torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len), generator=g),
             torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len), generator=g))
    return model, batch


def test_probe_runs_on_a_float64_copy_and_resolves_a_correct_adjoint() -> None:
    model, batch = _model_and_batch()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state()
    record: dict = {}
    worst = _fd_gradient_check(model, [batch], torch.device("cpu"), record=record)
    assert record["fd_gradient_dtype"] == "float64" and record["fd_gradient_eps"] == 1e-5
    assert math.isfinite(worst) and worst < 1e-5, (worst, record)
    assert math.isfinite(record["fd_gradient_noise_floor"]) and record["fd_gradient_noise_floor"] < 1e-6
    assert all(p.dtype == torch.float32 and p.grad is None for p in model.parameters())   # the live model untouched
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert torch.equal(rng, torch.get_rng_state())


def test_probe_flags_a_wrong_adjoint(monkeypatch: pytest.MonkeyPatch) -> None:
    model, batch = _model_and_batch()
    original = torch.Tensor.backward

    def scaled_backward(self: torch.Tensor, *args, **kwargs) -> None:            # a decode adjoint 10% too large
        original(1.1 * self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "backward", scaled_backward)
    worst = _fd_gradient_check(model, [batch], torch.device("cpu"), record={})
    assert abs(worst - 0.1 / 1.1) < 1e-4, worst
